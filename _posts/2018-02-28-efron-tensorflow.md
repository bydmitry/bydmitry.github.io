---
layout: post
title: Efron approximation in TensorFlow
tags: [time-to-event analysis, survival regression, tensorflow, coxph]
permalink: /efron-tensorflow.html
comments: true
---

_Efron's partial likelihood estimator is a method to handle tied events in Cox Survival Regression. Here we implement the method in TensorFlow to use it as an objective in a computational graph._

---

### Cox Proportional Hazards Recap

In Survival Analysis each observation $$i$$ can be expressed as a triple $$ (x_i, t_i. \delta_i) $$, where $$x_i$$ is a set of covariates, $$t_i > 0$$ is the follow-up time, and $$ \delta_i \in \{0,1\} $$ is the binary event indicator or censoring status. Cox Proportional Hazards model is probably one of the most popular methods to model time-to-event data. The model estimates the risk or probability of any arbitrary event of interest to occur at time _t_ given that the individual has survived until that time. This is called a __hazard function__ and in the Cox framework is expressed like this:

<div class="tex" align="center" data-expr="\lambda(t,x) = \lambda_0(t) exp(\beta^Tx)"></div>
<p/>

Here the first term is called the __baseline hazard__ and depends only on time. The baseline hazard remains unspecified, thus no particular distribution of survival times is assumed in the model. This is one of the reasons for the Cox model being so popular. The second term depends only on the covariates, but not time. This implies the __proportional hazard__ assumption, i.e. the effects of covariates on survival are constant over time.

### Estimation of the Cox Model
The full maximum likelihood requires that we specify the baseline hazard $$\lambda_0(t)$$. In 1972 Sir David Cox explained how to avoid having to specify this component explicitly and proposed a __partial likelihood__ that depends only on the parameters of interest, but not time:


<div class="tex" align="center" data-expr="L(\beta,x) = \prod_{i:\delta_i=1} \Bigg( \dfrac{e^{h_i}}{\sum_{j:t_j \ge t_i}  e^{h_j}} \Bigg)"></div>
<p/>

Here $$h_i = \beta^Tx_i$$ - predicted risk for individual $$i$$. This function iterates over subjects who fail and takes into account censored observations only in the risk set (denominator). The __problem__ with this likelihood is that it works only for continuous time survival data, which is obviously not the case in practical applications where multiple events may occur at the same (discrete) time. This complicates things a bit.

### Handling ties with Efron Approximation

A number of approaches have been suggested to handle tied events including the exact method that basically considers all possible orders of events that occurred at the same time. The exact approach, however, becomes computationally intensive as the number of ties grows. More efficient approaches have been proposed by Breslow and Efron. Efron's method though is considered to give better approximation to the original partial likelihood and is the default for R _survival_ package:

<div class="tex" align="center" data-expr="L(\beta,x) = \prod_{i:\delta_i=1} \Bigg( \dfrac{\prod_{j \in H_i} e^{h_j}} {\prod_{l=0}^{m-1} \Big( \sum_{j:t_j \ge t_i} e^{h_j} - \frac{l}{m} \sum_{j \in H_i} e^{h_j} \Big)} \Bigg) "></div>
<p/>

Here $$H_i$$ is a set of individuals that failed at time $$i$$.


#### An example

To understand what is going on here, let's consider an example. Say we observe a population with the following lifetimes and some risk value predicted for each individual:

```python
observed_times = np.array([5,1,3,7,2,5,4,1,1])
censoring      = np.array([1,1,0,1,1,1,1,0,1])
predicted_risk = np.array([0.1,0.4,-0.2,0.2,-0.3,0.0,-0.1,0.3,-0.4])
```

<div align="center">
<img src="/assets/img/figures/efron-tf-1.png">
</div>

So these are 9 observations whit red lines indicating the lifespan of individuals who experienced and event ($$\delta_i = 1 $$) and blue lines denote right-censored ($$\delta_i = 0 $$) cases. Here we have ties at $$t = 1 $$ and $$t = 5 $$.


As we typically like to optimize the __logarithm of the likelihood function__, let's rewrite Efron's formula again and give names to the key terms:

<div class="tex" align="center" data-expr="l(\beta,x) = \displaystyle\sum_{i:\delta_i=1} \Bigg(  \textcolor{#FF7741}{\underbrace{\sum_{j \in H_i} h_j}_{\text{tie\_risk}}}  -
    \sum_{l=0}^{\overbrace{m}^{\text{tie\_count}}-1} \log \Big( \textcolor{#7D3098}{\underbrace{\sum_{j:t_j \ge t_i} e^{h_j}}_{\text{cum\_hazard}}} - ({l}/{m}) \textcolor{#35B735}{\underbrace{\sum_{j \in H_i} e^{h_j}}_{\text{tie\_hazard}}} \Big) \Bigg)"></div>
<p/>

[//]: <> <div class="tex" align="center" data-expr="\log(L(\beta,x)) = \displaystyle\sum_{i:\delta_i=1} \Bigg( \textcolor{#FF7741}{\sum_{j \in H_i} h_j} -  \sum_{l=0}^{m-1} \log \Big( \textcolor{#7D3098}{\sum_{j:t_j \ge t_i} e^{h_j}} - ({l}/{m}) \textcolor{#35B735}{\sum_{j \in H_i} e^{h_j}} \Big) \Bigg)"></div>
[//]: <> <p/>

Using our toy dataset we can break down the equation into pieces and construct a table [first loop]. Note that the table is sorted by observed lifespans.

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;border-color:#ccc;border-width:1px;border-style:solid;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:0px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:0px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f0f0f0;}
.tg .tg-baqh{text-align:center;vertical-align:top}
.tg .tg-6nlm{color:#9b9b9b;text-align:center;vertical-align:top}
.tg .tg-i81m{background-color:#ffffff;text-align:center;vertical-align:top}
.tg .tg-0ql5{background-color:#ffffff;color:#9b9b9b;text-align:center;vertical-align:top}
</style>
<table class="tg">
  <tr>
    <th class="tg-baqh tex" width="10%" data-expr="i"></th>
    <th class="tg-baqh tex" width="10%" data-expr="t"></th>
    <th class="tg-baqh tex" width="10%" data-expr="\delta"></th>
    <th class="tg-baqh tex" width="17.5%" data-expr="h"></th>
    <th class="tg-baqh tex" width="17.5%" data-expr="\textcolor{#FF7741}{\sum_{j \in H_i} h_j}"></th>
    <th class="tg-baqh tex" width="17.5%" data-expr="\textcolor{#35B735}{\sum_{j \in H_i} e^{h_j}}"></th>
    <th class="tg-baqh tex" width="17.5%" data-expr="\textcolor{#7D3098}{\sum_{j:t_j \ge t_i} e^{h_j}}"></th>
  </tr>
  <tr>
    <td class="tg-i81m">5</td>
    <td class="tg-i81m">7</td>
    <td class="tg-i81m">1</td>
    <td class="tg-i81m">0.2</td>
    <td class="tg-i81m">0.2</td>
    <td class="tg-i81m">1.22</td>
    <td class="tg-i81m">1.22</td>
  </tr>
  <tr>
    <td class="tg-baqh" rowspan="2">4</td>
    <td class="tg-baqh">5</td>
    <td class="tg-baqh">1</td>
    <td class="tg-baqh">0.1</td>
    <td class="tg-baqh" rowspan="2">0.1</td>
    <td class="tg-baqh">2.105</td>
    <td class="tg-6nlm">2.33</td>
  </tr>
  <tr>
    <td class="tg-baqh">5</td>
    <td class="tg-baqh">1</td>
    <td class="tg-baqh">0.0</td>
    <td class="tg-baqh"></td>
    <td class="tg-baqh">3.33</td>
  </tr>
  <tr>
    <td class="tg-baqh">3</td>
    <td class="tg-baqh">4</td>
    <td class="tg-baqh">1</td>
    <td class="tg-baqh">-0.1</td>
    <td class="tg-baqh">-0.1</td>
    <td class="tg-baqh">0.905<br></td>
    <td class="tg-baqh">4.23</td>
  </tr>
  <tr>
    <td class="tg-i81m"></td>
    <td class="tg-i81m">3</td>
    <td class="tg-i81m">0</td>
    <td class="tg-i81m">-0.2</td>
    <td class="tg-i81m"></td>
    <td class="tg-i81m"></td>
    <td class="tg-0ql5">5.05</td>
  </tr>
  <tr>
    <td class="tg-i81m">2</td>
    <td class="tg-i81m">2</td>
    <td class="tg-i81m">1</td>
    <td class="tg-i81m">-0.3</td>
    <td class="tg-i81m">-0.3</td>
    <td class="tg-i81m">0.741</td>
    <td class="tg-i81m">5.79</td>
  </tr>
  <tr>
    <td class="tg-i81m" rowspan="3">1</td>
    <td class="tg-i81m">1</td>
    <td class="tg-i81m">1</td>
    <td class="tg-i81m">0.4</td>
    <td class="tg-i81m">0</td>
    <td class="tg-i81m">2.162</td>
    <td class="tg-0ql5">7.28</td>
  </tr>
  <tr>
    <td class="tg-i81m">1</td>
    <td class="tg-i81m">0</td>
    <td class="tg-i81m">0.3</td>
    <td class="tg-i81m"></td>
    <td class="tg-i81m"></td>
    <td class="tg-0ql5">8.63</td>
  </tr>
  <tr>
    <td class="tg-i81m">1</td>
    <td class="tg-i81m">1</td>
    <td class="tg-i81m">-0.4</td>
    <td class="tg-i81m"></td>
    <td class="tg-i81m"></td>
    <td class="tg-i81m">9.30</td>
  </tr>
  <tr>
    <th class="tg-baqh" width="10%">failure times</th>
    <th class="tg-baqh" width="10%">observed lifespans</th>
    <th class="tg-baqh" width="10%">censoring</th>
    <th class="tg-baqh" width="17.5%">predicted risk</th>
    <th class="tg-baqh" width="17.5%"><font color="#FF7741">tie_risk</font></th>
    <th class="tg-baqh" width="17.5%"><font color="#35B735">tie_hazard</font></th>
    <th class="tg-baqh" width="17.5%"><font color="#7D3098">cum_sum / cum_hazard</font></th>
  </tr>
</table>
<p/>

Now we just need to put the values together [second loop] :

$$\scriptsize i=5,m=1 \footnotesize: \textcolor{#FF7741}{0.2} - log(\textcolor{#7D3098}{1.22}-\textcolor{#35B735}{0}) = 0.00114914$$
$$\scriptsize i=4,m=2 \footnotesize: \textcolor{#FF7741}{0.1} - \big(  log(\textcolor{#7D3098}{3.33}-\textcolor{#35B735}{0}) + log(\textcolor{#7D3098}{3.33}-\textcolor{#35B735}{0.5*2.105}) \big) = -1.92605065$$
$$\scriptsize i=3,m=1 \footnotesize: \textcolor{#FF7741}{-0.1} - log(\textcolor{#7D3098}{4.23}-\textcolor{#35B735}{0}) = -1.54220199$$
$$\scriptsize i=2,m=1 \footnotesize: \textcolor{#FF7741}{-0.3} - log(\textcolor{#7D3098}{5.79}-\textcolor{#35B735}{0}) = -2.05613229$$
$$\scriptsize i=1,m=2 \footnotesize: \textcolor{#FF7741}{0.0} - \big( log(\textcolor{#7D3098}{9.30}-\textcolor{#35B735}{0}) + log(\textcolor{#7D3098}{9.30}-\textcolor{#35B735}{0.5*2.162}) \big) = -4.33646294$$
<p/>
$$\scriptsize 0.00114914 - 1.92605065 - 1.54220199 - 2.05613229 - 4.33646294 = \footnotesize \boxed{-9.859698}$$
<p/>


#### Implementation

We are going to store intermediate results in the following tensorflow variables:
```python
tie_count   = tf.Variable([], dtype=tf.uint8,   trainable=False)
tie_risk    = tf.Variable([], dtype=tf.float32, trainable=False)
tie_hazard  = tf.Variable([], dtype=tf.float32, trainable=False)
cum_hazard  = tf.Variable([], dtype=tf.float32, trainable=False)
```
These arrays are gathered while looping over individual failure times $$i$$ for the first time. Before we actually do the looping, we need to:

1). order the observation according to the observed lifespans:

```python
def efron_estimator_tf(time, censoring, prediction):
    n        = tf.shape(time)[0]
    sort_idx = tf.nn.top_k(time, k=n, sorted=True).indices
    risk     = tf.gather(prediction, sort_idx)
    events   = tf.gather(censoring, sort_idx)
    otimes   = tf.gather(time, sort_idx)
```

2). obtain unique failure times such that $$t_i > 0$$:

```python
otimes_cens   = otimes * events
unique_ftimes = tf.boolean_mask(otimes_cens, tf.greater(otimes_cens,0))
unique_ftimes = tf.unique(unique_ftimes).y
m = tf.shape(unique_ftimes)[0] # number of unique failure times
```

Now we are ready for looping. Indexing is the only part the might seem tricky because of type conversions. The code below is one step of the loop. First, we count number ties for each failure time:

```python
idx_b = tf.logical_and(
    tf.equal(otimes, unique_ftimes[i]),
    tf.equal(events, tf.ones_like(events)) )

tie_count = tf.concat([tie_count, [tf.reduce_sum(tf.cast(idx_b, tf.uint8))]], 0)
```

Since `tie_risk` and `tie_hazard` ignore censored observations they can be gathered as follows:

```python
idx_i = tf.cast(
    tf.boolean_mask(
        tf.lin_space(0., tf.cast(n-1,tf.float32), n),
        tf.greater(tf.cast(idx_b, tf.int32),0)
    ), tf.int32 )

tie_risk   = tf.concat([tie_risk, [tf.reduce_sum(tf.gather(risk, idx_i))]], 0)
tie_hazard = tf.concat([tie_hazard, [tf.reduce_sum(tf.gather(tf.exp(risk), idx_i))]], 0)
```

Finally, `cum_hazard` takes into account censored observations, so indexing changes:
```python
# this line actually stays outside of the loop
cs = tf.cumsum(tf.exp(risk))

idx_i = tf.cast(
    tf.boolean_mask(
        tf.lin_space(0., tf.cast(n-1,tf.float32), n),
        tf.greater(tf.cast(tf.equal(otimes, unique_ftimes[i]), tf.int32),0)
    ), tf.int32 )

cum_hazard = tf.concat([cum_hazard, [tf.reduce_max(tf.gather( cs, idx_i))]], 0)
```

Once these four vectors are gathered we will have to make the while loop again to finally calculate the likelihood:
```python
def loop_2_step(i, tc, tr, th, ch, likelihood):
    l = tf.cast(tc[i], tf.float32)
    J = tf.lin_space(0., l-1, tf.cast(l,tf.int32)) / l
    Dm = ch[i] - J * th[i]
    likelihood = likelihood + tr[i] - tf.reduce_sum(tf.log(Dm))
    return i + 1, tc, tr, th, ch, likelihood

loop_2_out = tf.while_loop(
    loop_cond, loop_2_step,
    loop_vars = [i, tie_count, tie_risk, tie_hazard, cum_hazard, log_lik],
    shape_invariants = [i.get_shape(),tf.TensorShape([None]),tf.TensorShape([None]),tf.TensorShape([None]),tf.TensorShape([None]),log_lik.get_shape()]
)

log_lik = loop_2_out[-1]
```

[Complete TensorFlow code for the function is on GitHub](https://github.com/bydmitry/efron-tf/blob/master/efrontf.py){:target="blank"}. If you run it on our dummy data you get __9.8594456__.

__If you find mistakes or have suggestions how to optimize the code, I would be happy if you share those with me!__



### Validation with R Survival package

I also wrote a [script](https://github.com/bydmitry/efron-tf/blob/master/validation.py){:target="blank"} to validate that our TensorFlow implementation is correct and produces the same result as R _survival_ package.


<script>
  window.onload = function() {
      var tex = document.getElementsByClassName("tex");
      Array.prototype.forEach.call(tex, function(el) {
          katex.render(el.getAttribute("data-expr"), el);
      });
  };
</script>
