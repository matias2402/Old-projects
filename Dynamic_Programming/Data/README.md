A. Karlstrom, M. Palme, and I. Svensson, "A Dynamic Programming Approach to
Model the Retirement Behavior of Blue-Collar Workers in Sweden", Journal of
Applied Econometrics, Vol. 19. No. 6, 2004, pp. 795-807.

The file sparadata.txt contains 51371 observations.
 
index   is an identifier for the observation (each individual and year)
id2    	is an identifier for the individual
sex    	1=man 2=woman
year   	of observation
age    	of the individual for the given year
married	marital status
retire	yes=1, no=0
income	1/1000 SEK. This number is rounded (no decimal digits) due to
	confidentiality reasons
atp	is the "average pension points" used in the paper

The file surv.txt contains survival probabilities. Each line contains an age
and a number less than 1. This number is the conditional survival
probability, starting with age 30. That is, the first row contains the
conditional probability of surviving to age 31, given that someone is alive
at age 30. The last row contains the conditional probability for age 107.
Anyone who reaches 108 is assumed to die with probability 1 before reaching
age 109.