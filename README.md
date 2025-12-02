# Raven: Recursively Reflect Act Validate and Enhance

This is an implementation of the RAVEN approach. As reference application, this code implements a Finance analysis engine that can carry out complex analytic tasks.


*Features* of Agent

- Reflect uses OpenAI API 
- Act Uses yfinance for ticker data and local functions that implement basic quantitative analyses: mean, volatility, ratios, momentum price, and symbol lookup
- Value uses simpler local LLMs to ascertain the validity of the actions
- Enhance is implemented an additional OpenAI step

State of the world is kept and managed in a domain specific Object.



References
1. A survey on large language model based autonomous agents  https://arxiv.org/pdf/2308.11432 (Has benchmarks. Survey of domain-specific applications)
