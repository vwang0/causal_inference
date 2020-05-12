# -*- coding: utf-8 -*-
"""
Created on Mon May 11 13:03:49 2020

@author: Vin

AB Testing With Python -  Udacity's Course Project

"""
import math as mt
import numpy as np
import pandas as pd
from scipy.stats import norm

# Place this estimators into a dictionary for ease of use later
baseline = {"Cookies":40000,
            "Clicks":3200,
            "Enrollments":660,
            "CTP":0.08,
            "GConversion":0.20625,
           "Retention":0.53,
           "NConversion":0.109313}

# Scale The counts estimates
baseline["Cookies"] = 5000
baseline["Clicks"]=baseline["Clicks"]*(5000/40000)
baseline["Enrollments"]=baseline["Enrollments"]*(5000/40000)
print(baseline)

# Get the p and n we need for Gross Conversion (GC)
# and compute the Stansard Deviation(sd) rounded to 4 decimal digits.
GC={}
GC["d_min"]=0.01
GC["p"]=baseline["GConversion"]
#p is given in this case - or we could calculate it from enrollments/clicks
GC["n"]=baseline["Clicks"]
GC["sd"]=round(mt.sqrt((GC["p"]*(1-GC["p"]))/GC["n"]),4)
print(GC["sd"])


# Get the p and n we need for Retention(R)
# and compute the Stansard Deviation(sd) rounded to 4 decimal digits.
R={}
R["d_min"]=0.01
R["p"]=baseline["Retention"]
R["n"]=baseline["Enrollments"]
R["sd"]=round(mt.sqrt((R["p"]*(1-R["p"]))/R["n"]),4)
print(R["sd"])

# Get the p and n we need for Net Conversion (NC)
# and compute the Standard Deviation (sd) rounded to 4 decimal digits.
NC={}
NC["d_min"]=0.0075
NC["p"]=baseline["NConversion"]
NC["n"]=baseline["Clicks"]
NC["sd"]=round(mt.sqrt((NC["p"]*(1-NC["p"]))/NC["n"]),4)
NC["sd"]

# Get z-score critical value and Standard Deviations
def get_sds(p,d):
    sd1=mt.sqrt(2*p*(1-p))
    sd2=mt.sqrt(p*(1-p)+(p+d)*(1-(p+d)))
    x=[sd1,sd2]
    return x

# Inputs: required alpha value (alpha should already fit the required test)
# Returns: z-score for given alpha
def get_z_score(alpha):
    return norm.ppf(alpha)

# Inputs p-baseline conversion rate which is our estimated p and d-minimum detectable change
# Returns: Standard Deviations
def get_sds(p,d):
    sd1=mt.sqrt(2*p*(1-p))
    sd2=mt.sqrt(p*(1-p)+(p+d)*(1-(p+d)))
    sds=[sd1,sd2]
    return sds

# Inputs:sd1-sd for the baseline,sd2-sd for the expected change,alpha,beta,d-d_min,p-baseline estimate p
# Returns: the minimum sample size required per group according to metric denominator
def get_sampSize(sds,alpha,beta,d):
    n=pow((get_z_score(1-alpha/2)*sds[0]+get_z_score(1-beta)*sds[1]),2)/pow(d,2)
    return n

# Calculate Sample Size per Metric
GC["d"]=0.01
R["d"]=0.01
NC["d"]=0.0075

# Calculate Gross Conversion
# Let's get an integer value for simplicity
GC["SampSize"]=round(get_sampSize(get_sds(GC["p"],GC["d"]),0.05,0.2,GC["d"]))
GC["SampSize"]=round(GC["SampSize"]/baseline['CTP']*2)
print(GC["SampSize"])
# This means we need at least 25,835 cookies who click the Free Trial button - per group! That means that if we got 400 clicks out of 5000 pageviews (400/5000 = 0.08) -> So, we are going to need GC["SampSize"]/0.08 = 322,938 pageviews, again ; per group! Finally, the total amount of samples per the Gross Conversion metric is:

# Calculate Retention
# Getting a nice integer value
R["SampSize"]=round(get_sampSize(get_sds(R["p"],R["d"]),0.05,0.2,R["d"]))
R["SampSize"]=R["SampSize"]/baseline['CTP']/baseline['GConversion']*2
print(R["SampSize"])
# This means that we need 39,087 users who enrolled per group! We have to first convert this to cookies who clicked, and then to cookies who viewed the page, then finally to multipky by two for both groups.
# This takes us as high as over 4 million page views total, this is practically impossible because we know we get about 40,000 a day, this would take well over 100 days. This means we have to drop this metric and not continue to work with it because results from our experiment (which is much smaller) will be biased.

# Calculate Net Conversion
# Getting a nice integer value
NC["SampSize"]=round(get_sampSize(get_sds(NC["p"],NC["d"]),0.05,0.2,NC["d"]))
NC["SampSize"]=NC["SampSize"]/baseline['CTP']*2
print(NC["SampSize"])
# We are all the way up to 685,325 cookies who view the page. This is more than what was needed for Gross Conversion, so this will be our number. Assuming we take 80% of each days pageviews, the data collection period for this experiment (the period in which the experiment is revealed) will be about 3 weeks.

# Analyzing Collected Data
# Finally, the moment we've all been waiting for, after so much preparation we finally get to see what this experiment will prove! The data is presented as two spreadsheets. I will load each spreadshot into a pandas dataframe.

# Loading collected data
# we use pandas to load datasets
control=pd.read_csv("control_data.csv")
experiment=pd.read_csv("experiment_data.csv")
print(control.head())

"""
Sanity Checks 
First thing we have to do before even beginning to analyze this experiment's results is sanity checks. These checks help verify that the experiment was conducted as expected and that other factors did not influence the data which we collected. This also makes sure that data collection was correct.

We have 3 Invariant metrics::

Number of Cookies in Course Overview Page
Number of Clicks on Free Trial Button
Free Trial button Click-Through-Probability
Two of these metrics are simple counts like number of cookies or number of clicks and the third is a probability (CTP). We will use two different ways of checking whether these obsereved values are like we expect (if in fact the experiment was not damaged.
"""
pageviews_cont=control['Pageviews'].sum()
pageviews_exp=experiment['Pageviews'].sum()
pageviews_total=pageviews_cont+pageviews_exp
print ("number of pageviews in control:", pageviews_cont)
print ("number of Pageviewsin experiment:" ,pageviews_exp)


p=0.5
alpha=0.05
p_hat=round(pageviews_cont/(pageviews_total),4)
sd=mt.sqrt(p*(1-p)/(pageviews_total))
ME=round(get_z_score(1-(alpha/2))*sd,4)
print ("The confidence interval is between",p-ME,"and",p+ME,"; Is",p_hat,"inside this range?")

"""
Our observed  p^  is inside this range which means the difference in number of samples between groups is expected. So far so good, since this invariant metric sanity test passes!

Number of cookies who clicked the Free Trial Button We are going to address this count with the same strategy as before.
"""
clicks_exp = experiment['Clicks'].sum()
clicks_total = clicks_cont + clicks_exp

p_hat = round(clicks_cont / clicks_total, 4)
sd = mt.sqrt(p * (1 - p) / clicks_total)
ME = round(get_z_score(1 - (alpha / 2)) * sd, 4)
print("The confidence interval is between", p - ME, "and", p + ME, "; Is",
      p_hat, "inside this range?")

"""
We have another pass! Great, so far it still seems all is well with our experiment results. Now, for the final metric which is a probability.
"""
ctp_cont = clicks_cont / pageviews_cont
ctp_exp = clicks_exp / pageviews_exp
d_hat = round(ctp_exp - ctp_cont, 4)
p_pooled = clicks_total / pageviews_total
sd_pooled = mt.sqrt(p_pooled * (1 - p_pooled) *
                    (1 / pageviews_cont + 1 / pageviews_exp))
ME = round(get_z_score(1 - (alpha / 2)) * sd_pooled, 4)
print("The confidence interval is between", 0 - ME, "and", 0 + ME, "; Is",
      d_hat, "within this range?")

"""
Examining effect size 
The next step is looking at the changes between the control and experiment groups with regard to our evaluation metrics to make sure the difference is there, that it is statistically significant and most importantly practically significant (the difference is "big" enough to make the experimented change beneficial to the company).

Now, all that is left is to measure for each evaluation metric, the difference between the values from both groups. Then, we compute the confidence interval for that difference and test whether or not this confidence interval is both statistically and practically significant.

Gross Conversion A metric is statistically significant if the confidence interval does not include 0 (that is, you can be confident there was a change), and it is practically significant if the confidence interval does not include the practical significance boundary (that is, you can be confident there is a change that matters to the business.)
Important: The given spreadsheet lists pageviews and clicks for 39 days, while it only lists enrollments and payments for 23 days. So, when working with enrollments and payments we should notice using only the corresponding pageviews and clicks, and not all of them.
"""
# Count the total clicks from complete records only
clicks_cont = control["Clicks"].loc[control["Enrollments"].notnull()].sum()
clicks_exp = experiment["Clicks"].loc[
    experiment["Enrollments"].notnull()].sum()

enrollments_cont = control["Enrollments"].sum()
enrollments_exp = experiment["Enrollments"].sum()

GC_cont = enrollments_cont / clicks_cont
GC_exp = enrollments_exp / clicks_exp
GC_pooled = (enrollments_cont + enrollments_exp) / (clicks_cont + clicks_exp)
GC_sd_pooled = mt.sqrt(GC_pooled * (1 - GC_pooled) *
                       (1 / clicks_cont + 1 / clicks_exp))
GC_ME = round(get_z_score(1 - alpha / 2) * GC_sd_pooled, 4)
GC_diff = round(GC_exp - GC_cont, 4)
print("The change due to the experiment is", GC_diff * 100, "%")
print("Confidence Interval: [", GC_diff - GC_ME, ",", GC_diff + GC_ME, "]")
print(
    "The change is statistically significant if the CI doesn't include 0. In that case, it is practically significant if",
    -GC["d_min"], "is not in the CI as well.")

"""
According to this result there was a change due to the experiment, that change was both statistically and practically significant. We have a negative change of 2.06%, when we were willing to accept any change greater than 1%. This means the Gross Conversion rate of the experiment group (the one exposed to the change, i.e. asked how many hours they can devote to studying) has decreased as expected by 2% and this change was significant. This means less people enrolled in the Free Trial after due to the pop-up.

Net Conversion The hypothesis is the same as before just with net conversion instead of gross. At this point we expect the fraction of payers (out of the clicks) to decrease as well.
"""

#Net Conversion - number of payments divided by number of clicks
payments_cont = control["Payments"].sum()
payments_exp = experiment["Payments"].sum()

NC_cont = payments_cont / clicks_cont
NC_exp = payments_exp / clicks_exp
NC_pooled = (payments_cont + payments_exp) / (clicks_cont + clicks_exp)
NC_sd_pooled = mt.sqrt(NC_pooled * (1 - NC_pooled) *
                       (1 / clicks_cont + 1 / clicks_exp))
NC_ME = round(get_z_score(1 - alpha / 2) * NC_sd_pooled, 4)
NC_diff = round(NC_exp - NC_cont, 4)
print("The change due to the experiment is", NC_diff * 100, "%")
print("Confidence Interval: [", NC_diff - NC_ME, ",", NC_diff + NC_ME, "]")
print(
    "The change is statistically significant if the CI doesn't include 0. In that case, it is practically significant if",
    NC["d_min"], "is not in the CI as well.")

"""
Double check with Sign Tests 
In a sign test we get another angle at analyzing the results we got - we check if the trend of change we observed (increase or decrease) was evident in the daily data. We are goint to compute the metric's value per day and then count on how many days the metric was lower in the experiment group and this will be the number of succssesses for our binomial variable. Once this is defined we can look at the proportion of days of success out of all the available days.
"""

#let's first create the dataset we need for this:
# start by merging the two datasets
full = control.join(other=experiment,
                    how="inner",
                    lsuffix="_cont",
                    rsuffix="_exp")
#Let's look at what we got
full.count()

#now we only need the complete data records
full = full.loc[full["Enrollments_cont"].notnull()]
full.count()

# Perfect! Now, derive a new column for each metric, so we have it's daily values
# We need a 1 if the experiment value is greater than the control value=
x = full['Enrollments_cont'] / full['Clicks_cont']
y = full['Enrollments_exp'] / full['Clicks_exp']
full['GC'] = np.where(x < y, 1, 0)
# The same now for net conversion
z = full['Payments_cont'] / full['Clicks_cont']
w = full['Payments_exp'] / full['Clicks_exp']
full['NC'] = np.where(z < w, 1, 0)
full.head()

GC_x=full.GC[full["GC"]==1].count()
NC_x=full.NC[full["NC"]==1].count()
n=full.NC.count()
print("No. of cases for GC:",GC_x,'\n',
      "No. of cases for NC:",NC_x,'\n',
      "No. of total cases",n)

"""
 Building a Sign Test 
We can forget all about this part and just use an online sign test calculator, but for me that is just no fun - so I will implement the calculations behind it.
What we want to do after we count the amount of days in which the experiment group had a higher metric value than that of the control group, is to see if that number is likely to be seen again in a new experiment (significance). We assume the chance of a day like this is random (50% chance to happen) and then use the binomial distribution with  p=0.5  and the number of experiments (days) to tell us the probability of this happening according to a random chance.
So, according to the binomial distribution with  p=0.5  and  n= total number of days; we want to now the probability of  x  days being a success (higher metric value in experiment). Because we are doing a two-tailed test we want to double this probability and once we have we can call it the  p−value  and compare it to our  α . If the  p−value  is greater than the  α  the result is not significant and vice-versa.

p(successes)=n!x!/(n−x)! * p^x*(1−p)^n−x 
Recall that a  p−value  is the probability of observing a test statistic as or more extreme than that observed. If we observed 2 days like that, the  p−value  for the test is:  p−value=P(x<=2) . We only need to remember the following:
P(x<=2)=P(0)+P(1)+P(2) .
"""


#first a function for calculating probability of x=number of successes
def get_prob(x, n):
    p = round(
        mt.factorial(n) / (mt.factorial(x) * mt.factorial(n - x)) * 0.5**x *
        0.5**(n - x), 4)
    return p


#next a function to compute the pvalue from probabilities of maximum x
def get_2side_pvalue(x, n):
    p = 0
    for i in range(0, x + 1):
        p = p + get_prob(i, n)
    return 2 * p


print("GC Change is significant if", get_2side_pvalue(GC_x, n),
      "is smaller than 0.05")
print("NC Change is significant if", get_2side_pvalue(NC_x, n),
      "is smaller than 0.05")





"""
Reference:

https://medium.com/@robbiegeoghegan/implementing-a-b-tests-in-python-514e9eb5b3a1
https://www.kaggle.com/tammyrotem/ab-tests-with-python/data
https://towardsdatascience.com/a-b-testing-design-execution-6cf9e27c6559
https://towardsdatascience.com/the-math-behind-a-b-testing-with-example-code-part-1-of-2-7be752e1d06f
"""