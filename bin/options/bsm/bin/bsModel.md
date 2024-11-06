#### Black Scholes Model 


### Standard Brownian Motion: 

- A standard Brownian motion is a random process  $x = \{X_t : t \isin [0,\infin]\}$ with a state space $\real$ that satisfies the following properties. 

1. $X_0 = 0$ with probability 1. 
   1. 100% probability that the initial value of the process is 0.
2. $X$ has stationary increments. That is for $s, t \isin [0, \infin]$ with $s < t$, the random variable $X_t - X_s$ has the same distribution as $X_{t-s}$. 
   1. The increments of the process are independent and identically distributed.
3. $X$ has independent increments. Thast is for, $t_1, t_2, ..., t_n \isin [0, \infin)$ with $t_1 < t_2 < ... < t_n$, the random variables $X_{t_1}, X_{t_2}, ..., X_{t_n}$ are independent. 
   1. The increments of the process are independent.
4. $X_t$ is normally distributed with mean 0 and variance $t$. 
   1. The increments of the process are normally distributed with mean 0 and variance $t$.
5. With probability 1, $t \rightarrow X_t$ is continuous on $[0, \infin)$


## Demo of Brownian Motion

details found here: [Brownian Motion](https://towardsdatascience.com/brownian-motion-with-python-9083ebc46ff0)


## Geometric Brownian Motion (GBM) is a **continuious-time** stochastic process 

Suppose that $z = \{ Z_t: t \isin [0, \infin]\}$ is a Standard Brownian Motion, and that $\mu \isin \real \  \& \ \sigma \isin (0, \infin)$:

$$X_t = \exp\left[\left(\mu - \frac{\sigma^2}{2}\right) t + \sigma Z_t\right], \quad t \in [0, \infty)$$

The Stochastic Process $x = \{X_t : t \isin [0,\infin]\}$
- in which the logarithim of the randomly varying quantity follows a Browniuan Motion (Wiener proccess) with a drift. 
- Dynamics are controlled by the mean and variance parameters of the underlying Normal Distribution. This emulates the grownth and the 'volatility'  of the underlying stock. 

---

### To simulate the stock price: 

$$S_t = S_0 \ e^{(\mu - \frac{\sigma ^2}{2})t + \sigma W_t}, \quad t ≥ 0$$

- For example, a stock with a postive grownth trend will have a postive mean. For this particular simulation the choice of the mean is .2 and standard deviation is .68
  


# Monte carlo simulation on the Black-scholes model: 

[Page](https://kinder-chen.medium.com/black-scholes-model-and-monte-carlo-simulation-d8612ac4519b)

- The Black-Scholes model is a partial differential equation that is widely used to price option contracts.

**Europen Call Option:**

$$C(S,t) = N(d_1)S_t - N(d_2)Ke^{-rt} \\ \text{where: } d_1 =  (ln(\frac{S_t}{K}) + t(r+\frac{\sigma^2}{2})) \frac{1}{\sigma \sqrt{t}} \\ \text{and: } d_2 = d_1 - \sigma \sqrt{t}$$

-> The Seven Inputs Needed are: 
  - $C$ = Call option Price
  - $N$ = CDF of Normal Distribution 
  - $K$ = Strike Price of the contract 
  - $S_t$ = Spot Price of an asset
  - $r$ = risk-free interest rate
  - $t$ = time to maturity 
  - $\sigma$ = volatility of an asset
  - $e^{-rt}$ = Time Decay 

**European Put Option:**

$$P(S, t) = Ke^{-rt} - S_t + C(S_t, t) \\ = N(-d_2) K e^{-r(T-t)} - N(-d_1) S_t$$

$$\text{where: } d_1 = \frac{ (ln(\frac{S_t}{K}) + (T-t)(r+\frac{\sigma^2}{2})) }{\sigma \sqrt{(T-t)}} \\ \ \\ \text{and: } d_2 = d_1 - \sigma \sqrt{(T-t)}$$

---- 

The underlying stock follows the Goemetric Brownian Motion, and satisfies the following stochastic differential equation:
$$dS_t = \mu S_t dt + \sigma S_t dW_t$$

- With $\mu$ is the drift rate of grownth of the stock, and W denotes a Weiner process (one dimensional brownian motion).
- The analytic solution of the SDE can be solved as: 
$$S_t = S_0 e^{(\mu - \frac{\sigma^2}{2})t + \sigma \sqrt{t}Z}$$

- Where $Z$ is a standard normal random variable. 
- Model assumes prices follow a lognormal distribution. Asset prices are often observed to have significant right skewness have higher likelihood of being observed at lower asset prices. 


--- 
## Simulate the option price with **multiple sources of uncertainty**

1. First the price of the asset is sumulated by using random number generation for a number of paths. 
2. After repeatedly simulating tragectories and computing averages **The estimated price of options can be obtained which is consistendt with the analytical results from black scholes model.**


### For European Options the buyer elects to excersise the option on the maturity date. So the price at time $t=0$: 



$$\text{Calls: } C_0 = e^{-rT} \frac{1}{N} \sum^N_i{max(S_T - K, 0)}$$

$$\text{Puts: }P_0 = e^{-rT} \frac{1}{N} \sum^N_i{max(K - S_T, 0)}$$


----
__The Black Scholes model is a differential equation that is widely used to price option contracts.__

$$C = \phi(d_1)S_t - \phi(d_2)Ke^{-rt} \\ \text{where: } d_1 =  (ln(\frac{S_t}{K}) + t(r+\frac{\sigma^2}{2})) \frac{1}{\sigma \sqrt{t}} \\ \text{and: } d_2 = d_1 - \sigma \sqrt{t}$$

-> The Seven Inputs Needed are: 
- $C$ = Call option Price
- $\phi$ = CDF of Normal Distribution 
- $K$ = Strike Price of the contract 
- $S_t$ = Spot Price of an asset
- $r$ = risk-free interest rate
- $t$ = time to maturity 
- $\sigma$ = volatility of an asset
- $e^{-rt}$ = Time Decay 

-----

### Stock Price Assumptions: 

The underlying stock follows the Goemetric Brownian Motion, and satisfies the following stochastic differential equation:
$$dS_t = \mu S_t dt + \sigma S_t dW_t$$

- With $\mu$ is the drift rate of grownth of the stock, and W denotes a Weiner process (one dimensional brownian motion).
- The analytic solution of the SDE can be solved as: 
$$S_t = S_0 e^{(\mu - \frac{\sigma^2}{2})t + \sigma \sqrt{t}Z}$$

- Where $Z$ is a standard normal random variable. 
- Model assumes prices follow a lognormal distribution. Asset prices are often observed to have significant right skewness have higher likelihood of being observed at lower asset prices. 

- We also assume that interest rates are constant so that 1 unit of currency in the cash account at time 0 will be worth $B_t:=e^{rt}$ at time $t$
- The value of the Call Option at time $t$ is $C(S, t)$

$$
dC(S, t) = \bigl ( \frac{\partial C}{\partial t} + \mu S \frac{\partial C}{\partial S} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 C}{\partial S^2} \bigr ) dt + \sigma S \frac{\partial C}{\partial S} dW_t
$$

----

**Self Trading Portfolio:** There is no exogenus infusion or withdrawl of money, the purchase of an asset must be financed by the sale of an old one. 

- Let $h_i(t)$ be denote the number of shares for stock $i$ at time $t$ in the portfoliio. 
- $S_i(t)$ the price of stock $i$ at time $t$. (**Frictionless market**)
- $P(t)$ is the Value of the portfolio at time $t$.
$$
P(t) = \sum^n_{i = 1} h_i(t) S_i(t)
$$
- A **Self Financing Portfolio** is when the value of the portfolio at time $t$ is equal to the value of the portfolio at time $t-1$ plus the value of the new shares purchased at time $t$ minus the value of the old shares sold at time $t$.
$$
dP(t) = \sum^n_{i = 1} h_i(t) dS_i(t)
$$
- **Let $x_t$ be units of cash held in the account**
- **$y_t$ be the number of shares held in the account.

- Recall, We also assume that interest rates are constant so that 1 unit of currency in the cash account at time 0 will be worth $B_t:=e^{rt}$ at time $t$
- The value of the Call Option at time $t$ is $C(S, t)$

- Then the value of the portfolio at time $t$ is given by:
$$
P_t = x_t B_t + y_t S_t
$$
- We choose $x_t$ and $y_t$ in a way that the strategy replicates the value of the option. The self-financing implies that: 
$$
dP_t = x_t dB_t + y_t dS_t \\ \ \\
= r x_t B_t d_t + y_t (\mu S_t dt + \sigma S_t dW_t) \\ \ \\
= (rx_t B_t + \mu y_t S_t) dt + \sigma y_t S_t dW_t
$$

- Any gains or losses on the portfolio are due entirely to gains or losses in the **underlying security** 
  - The cash account and stock account are **uncorrelated** with changes in the holdings $x_t$ and $y_t$.
We can equate the number of shares held in the account to the partial derivative of the option price with respect to the stock price.
$$
y_t = \frac{\partial C}{\partial S} \\ \ \\
r x_t B_t = \frac{\partial C}{\partial S} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 C}{\partial S^2}
$$

If we set $C_0 = P_0$ the initial value of the self-financing strategy, then it must be the case that $C_t = P_t$ for all $t$, since $C$ and $P$ are both solutions to the Black-Scholes PDE. This is the principle of no-arbitrage. This is true by consturctions after equating the terms in the previous equation. We can substute to obtain the following equation:

$$
rS_t \frac{\partial C}{\partial S}+ \frac{\partial C}{\partial t} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 C}{\partial S^2} - rC = 0
$$
 
The solution for a call option is: 

$$C = \phi(d_1)S_t - \phi(d_2)Ke^{-rt} \\ \text{where: } d_1 =  (ln(\frac{S_t}{K}) + t(r+\frac{\sigma^2}{2})) \frac{1}{\sigma \sqrt{t}} \\ \text{and: } d_2 = d_1 - \sigma \sqrt{t}$$

If we set $\mu = r$ then this method of derivatives pricing can be known as **Risk Neutral Pricing**
----

**Martingale Pricing** is pricing approach based on the notions of martingale and risk neutrality.

- In probability **Martingale** is a sequence of random variables (*a stochastic process*) **for which at a particular time the conditional expectation of the next value in the sequence is eqal to the present value, regardless of all prior values.**
$$\text{Discrete: }E(|Y_{n +1}| X_1, ..., X_n) = Y_n \\ \ \\
\text{Continuous: }E(|Y_t| \{X_{\Tau}, \Tau ≤ s \}) =  Y_s \ \ \forall s≤ t$$
- An unbiased random walk is an example of a martingale. 


- It can be shown that Black Scholes PDE in is consitstent with martingale pricing. In particular if we deflate by the cash account then the deglated stock price process. $Y_t := \frac{S_t}{B_t}$ must be a Q-martingale. Where $Q$ is the EMM (Equivalent Martingale Measure) and as numerarie. It can be shown that $Q$ dynamics of $S_t$ satisfy: 

$$ 

dS_t = r S_t dt + \sigma S_t dW_t^Q \\ \ \\
\text{This Implies: } S_T = S_t e^{(r - \frac{\sigma^2}{2})(T-t) + \sigma (W_T^Q - W_t^Q)} \\ \ \\

$$
 
- So that $S_t$ is log-normally distributed under $Q$ It is now easily confirmed that the call option price also satisfies the Black Scholes PDE.

$$
C(S_t, t) = E^Q_t[e^{-r(T-t)} \max(S_t - K, 0)]


$$
##### Volatility Surface: 

$$
C(S, K, T) := BS(S, T, K, r, q, K, \sigma(K, T))
$$

- Where $C(S, K, T)$ is the current **market-price** of a call option with time to maturity $T$ and strike $K$. 
- $BS(\cdot)$  is the Black-Scholes formula for pricing a a call option
- $\sigma(K, T)$ is the volatility that when substituted into the BS model, returns the market price $C(S, K, T)$