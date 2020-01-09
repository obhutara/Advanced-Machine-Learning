###yaml setup for lab 1###
###LAB 1 stuff###

---
  title: "Advanced Machine Learning Lab 1 (Graphical Models)"
author: "Omkar Bhutra (omkbh878)"
date: "21 September 2019"
output: pdf_document
---
  
```{r lib, message=FALSE,error=FALSE,warning=FALSE,echo=FALSE}
knitr::opts_chunk$set(echo = TRUE)
library("dplyr")
library("ggplot2")
library("gRain")
library("bnlearn")
```

#Task 1:
```{r 1, message=FALSE,error=FALSE,warning=FALSE}
data("asia")
asia <- as.data.frame(asia)
#default parameter run of HC
hillClimbingResults = hc(asia)
print(hillClimbingResults)
par(mfrow = c(2,3))

hillclimb <- function(x){
  hc1 <- hc(x,restart = 15,score = "bde", iss = 3)
  hc2 <- hc(x,restart = 10,score = "bde", iss = 5)
  hc1dag <- cpdag(hc1)
  plot(hc1dag, main = "plot of BN1")
  hc1arcs <- vstructs(hc1dag, arcs = TRUE)
  # print(hc1arcs)
  arcs(hc1)
  hc2dag <- cpdag(hc2)
  plot(hc2dag, main = "plot of BN2")
  hc2arcs <- vstructs(hc2dag, arcs = TRUE)
  # print(hc2arcs)
  arcs(hc2)
  print(all.equal(hc1dag,hc2dag))
}
for(i in 1:3){
  hillclimb(asia)
}
```

Non-equivalant solutions are produced using different starting parameters such as number of restarts, scoring algorithm, imaginary sample size etc. Hill climbing algorithm is a deterministic one, and hence generates different results on different input parameters. Hill climbing leads to a local maxima of the objective function which makes two different results possible with the same code. Adding imaginary sample size affects the relationships between the nodes such that we get more edges. Number of restarts increases the possibility of getting the same result.


#Task 2:
Network structure is trained by the hill climbing algorithm using the BDE score. bn.fit is used to learn the paramets using maximum liklihood estimation. To predict S in the test data , evidence in the network is set to the values of the test data excluding S, using setEvidence. querygrain is used to find the probabilty, maximum of the values is taken to find the misclassification rate.

```{r 2, message=FALSE,error=FALSE,warning=FALSE}
#setting up training and testing datasets
id <- sample(x = seq(1, dim(asia)[1], 1), 
             size = dim(asia)[1]*0.8,
             replace = FALSE)
asia.train <- asia[id,]
asia.test <- asia[-id,]

bnprediction = as.numeric()
#fitting a model using Hill Climbing
bnmodel <- hc(asia.train,score = "bde",restart = 50)
bnmodelfit <- bn.fit(bnmodel,asia.train,method = 'mle') #max liklihood estimation
bngrain <- as.grain(bnmodelfit)
comp <- compile(bngrain)

bnmodel_true <- model2network("[A][S][T|A][L|S][B|S][D|B:E][E|T:L][X|E]")
bnmodelfit_true <- bn.fit(bnmodel_true, asia.train)
bngrain_true <- as.grain(bnmodelfit_true)
comp_true <- compile(bngrain_true)

bnpredict <- function(bntree, data, obs_variables, pred_variable) {
  for (i in 1:dim(data)[1]) {
    x <- NULL
    for (j in obs_variables) {
      x[j] <- if(data[i, j] == "yes") "yes" else "no"
    }
    evidence <- setEvidence(object = bntree,nodes = obs_variables,states=x)
    prob_dist_fit  <- querygrain(object = evidence,nodes = pred_variable)$S
    bnprediction[i] <- if (prob_dist_fit["yes"] >= 0.5) "yes" else "no"
  }
  return(bnprediction)
}

# Predict S from Bayesian Network and test data observations
bnprediction <- bnpredict(bntree = comp,
                          data = asia.test,
                          obs_variables = c("A", "T", "L", "B", "E", "X", "D"),
                          pred_variable = c("S"))
bnprediction_true <- bnpredict(bntree = comp_true,
                               data = asia.test,
                               obs_variables = c("A", "T", "L", "B", "E", "X", "D"),
                               pred_variable = c("S"))

# Calculate confusion matricies
confusion_matrix_fit <- table(bnprediction, asia.test$S)
print(confusion_matrix_fit)
print(paste("Misclassification rate:", 1-sum(diag(confusion_matrix_fit))/sum(confusion_matrix_fit)))

confusion_matrix_true <- table(bnprediction_true, asia.test$S)
print(confusion_matrix_true)
print(paste("Misclassification rate:", 1-sum(diag(confusion_matrix_true))/sum(confusion_matrix_true)))

par(mfrow=c(1,2))
plot(bnmodel)
title("hill climb network")
plot(bnmodel_true)
title("True network")
par(mfrow=c(1,1))
```
The Misclassification rate remain the same for both the hill climb network and the true network. However, The state of "Different number of directed/undirected arcs" persists.

#Task 3:
Using the markov blanket, we predict the results again, it is expected to output the same result. 

```{r 3, message=FALSE,error=FALSE,warning=FALSE}
markov_blanket = mb(bnmodel, c("S"))
markov_blanket_true = mb(bnmodel_true, c("S"))
bnpredict_mb <- bnpredict(comp, asia.test, markov_blanket, c("S"))
bnpredict_mb_true <- bnpredict(comp_true, asia.test, markov_blanket_true, c("S"))
confusion_matrix_fit <- table(bnpredict_mb, asia.test$S)
confusion_matrix_fit_true <- table(bnpredict_mb_true, asia.test$S)
print(confusion_matrix_fit)
print(confusion_matrix_fit_true)
print(paste("Misclassification rate:", 1-sum(diag(confusion_matrix_fit))/sum(confusion_matrix_fit)))
print(paste("Misclassification rate:", 1-sum(diag(confusion_matrix_fit_true))/sum(confusion_matrix_fit_true)))
```
#Task 4:
The naiveBayes structure implies that all variables are independant of S, this implies that there is a diffrent true structure used.
```{r 4, message=FALSE,error=FALSE,warning=FALSE}
naiveBayes = model2network("[S][A|S][B|S][D|S][E|S][X|S][L|S][T|S]")
plot(naiveBayes, main = "Naive Bayes")

naiveBayes.fit <- bn.fit(naiveBayes, asia.train)
result_naive <- bnpredict(compile(as.grain(naiveBayes.fit)), asia.test, c("A", "T", "L", "B", "E", "X", "D"), c("S"))
# Calculate confusion matricies
confusion_matrix_naive_bayes <- table(result_naive, asia.test$S)
print(confusion_matrix_naive_bayes)
print(paste("Misclassification rate:", 1-sum(diag(confusion_matrix_naive_bayes))/sum(confusion_matrix_naive_bayes)))

```
#Task 5:
In Task 2, Same results are observed for the trained structure and the true structure. Same results are also obtained in Task 3 as before, this is expected due to the reduction of the node from the rest of the structure. The reduced number of predictors still provides the same misclassification rate although some information loss is expected.
In Task 4, Naive bayes is not a predictor for the data as the structure used is different as the variables are inferred to be independant of S, but all are assumed to have an effect on S.

######LAB 2 stuff###

---
  title: "Advance Machine Learning Lab 2"
author: "Omkar Bhutra (omkbh878)"
date: "26 September 2019"
output: 
  pdf_document:
  latex_engine: xelatex
---
  
  ```{r lib, message=FALSE,error=FALSE,warning=FALSE,echo=FALSE}
knitr::opts_chunk$set(echo = TRUE)
library("dplyr")
library("ggplot2")
library("gRain")
library("bnlearn")
library("HMM")
library("entropy")
```

The purpose of the lab is to put in practice some of the concepts covered in the lectures.
To do so, you are asked to model the behavior of a robot that walks around a ring. The ring is
divided into 10 sectors. At any given time point, the robot is in one of the sectors and decides
with equal probability to stay in that sector or move to the next sector. You do not have direct
observation of the robot. However, the robot is equipped with a tracking device that you can
access. The device is not very accurate though: If the robot is in the sector i, then the device
will report that the robot is in the sectors [i - 2; i + 2] with equal probability.

###Question 1: Build a hidden Markov model (HMM) for the scenario described above 

A robot moves around a ring which divided into 10 sectors. The robot is in one sector at any given
time step and its equally probable for the robot to stay in the state as it is to move to the next state.
The robot has a tracking device. If the robot is in sector i, the tracking device will report that the robot is in the sectors [i - 2, i + 2] with equal probability i.e P = 0.2 for being in each position of that range.

Create transition matrix where each row consists of: $P(Z^t|Z^t-1),t = 1, ..., 10$

```{r 1, message=FALSE,error=FALSE,warning=FALSE}
states <- paste("state",1:10,sep="")
symbols <- paste("symbol",1:10,sep="")
transition_vector <- c(0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5,
0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0.5)

emission_vector <- c(0.2, 0.2, 0.2, 0, 0, 0, 0, 0, 0.2, 0.2,
0.2, 0.2, 0.2, 0.2, 0, 0, 0, 0, 0, 0.2,
0.2, 0.2, 0.2, 0.2, 0.2, 0, 0, 0, 0, 0,
0, 0.2, 0.2, 0.2, 0.2, 0.2, 0, 0, 0, 0,
0, 0, 0.2, 0.2, 0.2, 0.2, 0.2, 0, 0, 0,
0, 0, 0, 0.2, 0.2, 0.2, 0.2, 0.2, 0, 0,
0, 0, 0, 0, 0.2, 0.2, 0.2, 0.2, 0.2, 0,
0, 0, 0, 0, 0, 0.2, 0.2, 0.2, 0.2, 0.2,
0.2, 0, 0, 0, 0, 0, 0.2, 0.2, 0.2, 0.2,
0.2, 0.2, 0, 0, 0, 0, 0, 0.2, 0.2, 0.2)


transition_matrix <- matrix(data = transition_vector,nrow = 10,ncol = 10)

emission_matrix <- matrix(data = emission_vector,nrow = 10,ncol = 10)
# States are the hidden variables
# Symbols are the observable variables
hmm <- initHMM(States = states,
Symbols = symbols,
startProbs = rep(0.1,10),
transProbs = transition_matrix,
emissionProbs = emission_matrix)
hmm
```

###Question 2: Simulate the HMM for 100 time steps.

```{r 2, message=FALSE,error=FALSE,warning=FALSE}
hmm_sim <- simHMM(hmm = hmm,length = 100)
hmm_sim
```

###Question 3: Discard the hidden states from the sample obtained above. Use the remaining observationsto compute the filtered and smoothed probability distributions for each of the 100 time points. Compute also the most probable path. Question 4: Compute the accuracy of the filtered and smoothed probability distributions, and of themost probable path. That is, compute the percentage of the true hidden states that are guessed by each method
Hint: Note that the function forward in the HMM package returns probabilities in log
scale. You may need to use the functions exp and prop.table in order to obtain a
normalized probability distribution. You may also want to use the functions apply and
which.max to find out the most probable states. Finally, recall that you can compare
two vectors A and B elementwise as A==B, and that the function table will count the
number of times that the different elements in a vector occur in the vector.

Forward probabilities can be found in a hidden markov model with hidden states X upto observation at time t defined as the probability of observing the sequence of observations $e_1, ... ,e_k$ and that the state at time t is X. That is: $f[X,t] := P(X|E_1 = e_1, ... , E_k = e_k)$

```{r 3-4, message=FALSE,error=FALSE,warning=FALSE}
robot <- function(hiddenmarkovmodel,pars){
hmm_sim <- simHMM(hmm = hmm,length = 100)
hmm_obs <- hmm_sim$observation
hmm_states <- hmm_sim$states
#filter: forward function does the filtering and returns log probabilities
log_filter = forward(hmm = hmm,observation = hmm_obs)
filter = exp(log_filter)
#normalised probability distribution
filternormalised <- prop.table(filter,margin = 2)
# find out the most probable states
filternormalised_probable <- apply(filternormalised,MARGIN = 2, FUN = which.max)
accuracy_filtering <- sum(paste("state", filternormalised_probable, sep ="")
==hmm_states)/length(hmm_states)
#filternormalised
#accuracy_filtering
#smoothing using function posterior
smooth <- posterior(hmm,hmm_obs)
smoothnormalised <- prop.table(smooth, margin = 2)
smoothnormalised_probable <- apply(smoothnormalised,MARGIN = 2, FUN = which.max)
accuracy_smooth <- sum(paste("state", smoothnormalised_probable, sep ="")
==hmm_states)/length(hmm_states)
#smoothnormalised
#accuracy_smooth
#Finding the most probable path using viterbi algorithm
probable_path <- viterbi(hmm = hmm, observation = hmm_obs)
accuracy_probable_path <- sum(probable_path == hmm_states)/ length(hmm_states)
probable_path #most probable path
#accuracy_probable_path
if(pars == "filter"){
return(filternormalised)
}
if(pars == "smooth"){
return(smoothnormalised)
}
if(pars == "ProbablePath"){
return(probable_path)
}
if(pars == "accuracy"){
return(c(accuracy_filtering = accuracy_filtering,
accuracy_smooth = accuracy_smooth,
accuracy_probable_path = accuracy_probable_path))
}
}  
```

###Question 5: Repeat the previous exercise with different simulated samples. In general, the smoothed distributions should be more accurate than the filtered distributions. Why ? In general,the smoothed distributions should be more accurate than the most probable paths, too. Why ?
```{r 5.0, message=FALSE,error=FALSE,warning=FALSE,echo=FALSE, include=FALSE}
robot(hmm,"filter")
robot(hmm,"smooth")
```

```{r 5.1, message=FALSE,error=FALSE,warning=FALSE}
robot(hmm,"ProbablePath")

acc <- sapply(1:100,FUN = function(x){robot(hmm,"accuracy")})
acc <- data.frame(t(acc))
acc$number <- 1:100

ggplot(data = acc) + geom_line(aes(x=number,y=accuracy_filtering,col="filtered"))+
geom_line(aes(x=number,y=accuracy_smooth,col="smoothened"))+
geom_line(aes(x=number,y=accuracy_probable_path,col="Probable Path"))

colMeans(acc[,-4])
```

The smoothed distribution is using more information i.e the whole set of observed emission x[0:T] whereas filtered only uses data emitted up to that point x[0:t]
The most probable path generated by the viterbi algorithm is constrained by the transitions between states in the hidden variables. The smooth distribution approximates the most probable state and hence can make jumps from one state to another when predicting current state.

### Question 6: Is it true that the more observations you have the better you know where the robot is ?
Hint: You may want to compute the entropy of the filtered distributions with the function
entropy.empirical of the package entropy.

```{r 6, message=FALSE,error=FALSE,warning=FALSE}
hmm_filter <- robot(hmm,"filter")
hmm_filter_entropy <- data.frame(index = 1:100, Entropy = apply(hmm_filter, MARGIN = 2,
FUN = entropy::entropy.empirical))
ggplot(hmm_filter_entropy,aes(x = index, y = Entropy))+
geom_line()+ggtitle("Entropy of filtered distribution")
```
The entropy is random even when we increase the number of observations added to the hidden markov model. This is because the HMM is Markovian and only depends on the previous observation.

###Question 7: Consider any of the samples above of length 100. Compute the probabilities of the hidden states for the time step 101.

```{r 7, message=FALSE,error=FALSE,warning=FALSE}
posterior <- hmm_filter[,100]
# matrix multiplication
probability <- as.data.frame(hmm$transProbs %*% posterior)
ggplot(probability)+geom_line(aes(x=1:10,y=V1))+ggtitle("probability of the hidden state and symbol for the timestep 101")+labs(x="states",y="probability")
```
The Markovian assumption is that only relavant state is the current state. All previous states feeds though the 100 states and provides us information about the states and observations. On multiplication with transition probabilities to get the probability of the hidden state and symbol for the timestep 101.
That the entropy of the filtered distributions does not decrease monotonically. It has to do with the fact that you get noisy observations and, thus, you will more often than not be uncertain as to where the robot is.


###Lab 3 stuff###

---
  title: "Advance Machine Learning Lab 3"
author: "Omkar Bhutra (omkbh878), Obaid Ur Rehman (obaur539) ,Ruben Jared MuÃ±oz Roquet (rubmu773)"
date: "7 October 2019"
output: pdf_document
---
  
  ```{r lib, message=FALSE,error=FALSE,warning=FALSE,echo=FALSE}
knitr::opts_chunk$set(echo = TRUE)
library("dplyr")
library("ggplot2")
library("KFAS")
```

#Q1. The purpose of the lab is to put in practice some of the concepts covered in the lectures. To do so, you are asked to implement the particle filter for robot localization. For the particle filter algorithm, please check Section 13.3.4 of Bishops book and/or the slides for the last lecture on state space models (SSMs). The robot moves along the horizontal axis according to the following SSM:

Transition Model: $p(z_t|z_{t-1}) = (N(z_t|z_{t-1},1)+N(z_t|z_{t-1}+1,1)+N(z_t|z_{t-1}+2,1))/3$
  
  Emission Model: $p(x_t|z_t) = (N(x_t|z_t,1)+N(x_t|z_t-1,1)+N(x_t|z_t+1,1))/3$
  
  Initial Model: $p(z_1) = Uniform(0,100)$
  
  ##A) Implement the SSM above. Simulate it for T = 100 time steps to obtain z 1:100 (i.e., states) and x 1:100 (i.e., observations). Use the observations (i.e., sensor readings) to identify the state (i.e., robot location) via particle filtering. Use 100 particles. Show the particles, the expected location and the true location for the first and last time steps, as well as for two intermediate time steps of your choice.
  
  
  ```{r 1, message=FALSE,error=FALSE,warning=FALSE}
initModel <- function(len){
  x <- runif(len,0,100)
  return(x)
}

transitionmodel <- function(zt){
  probs = rep(1/3,3)
  draw = sample(1:3,1,prob = probs)
  if(draw==1){
    normTrans <- rnorm(1,zt,transition_sd)
  }
  else if(draw==2){
    normTrans <- rnorm(1,zt+1,transition_sd)
  }
  else{
    normTrans <- rnorm(1,zt+2,transition_sd)
  }
  return(normTrans)
}

emmisionmodel <- function(zt,emission_sd){
  probs = rep(1/3,3)
  draw = sample(1:3,1,prob = probs)
  
  if(draw==1){
    normEmis <- rnorm(1,zt,emission_sd)
  }
  else if(draw==2){
    normEmis <- rnorm(1,zt-1,emission_sd)
  }
  else{
    normEmis <- rnorm(1,zt+1,emission_sd)
  }
  return(normEmis)
}

emission_density <- function(xt,zt){
  x <- (dnorm(xt,zt,emission_sd)+dnorm(xt,zt-1,emission_sd)+dnorm(xt,zt+1,emission_sd))/3
  return(x)
}


emission_sd = 1
transition_sd = 1

ssm <- function(emission_sd,transition_sd){
  T = 100
  particles = 100
  zt = xt = error = rep(NA,T)
  
  # for zt and xt
  zt[1] = initModel(1)
  for(i in 2:T){
    zt[i] <- transitionmodel(zt[i-1])
    xt[i] <- emmisionmodel(zt[i],emission_sd)
  }
  
  initParticles <- initModel(particles)
  particleWeights<- rep(1/particles,particles)
  estimation <- c()
  Particles25 = Particles75 = NA
  
  for(i in 2:T){
    newParticles <- sample(1:particles,particles,prob=particleWeights,replace = T)
    initParticles <- sapply(initParticles[newParticles],transitionmodel)
    
    if(i == 25){
      Particles25 <- c(initParticles)
    }
    else if(i == 75){
      Particles75 <- c(initParticles)
    }
    
    
    for(j in 1:particles){
      particleWeights[j] <- emission_density(xt[i],initParticles[j])
    }
    #Normalise
    particleWeights <- particleWeights/sum(particleWeights)
    
    Ezt <- sum(particleWeights*initParticles)
    error[i] <- abs(zt[i]-Ezt)
    estimation[i] <- sum(particleWeights * initParticles)
  }
  
  data = data.frame(zt,xt,estimation,particles = initModel(particles),Particles25,Particles75,initParticles)
  
  ggplot(data,aes(x=c(1:T),y=value,color=variable,xlab="timestep"))+
    geom_line(aes(y=data$zt,col='True Zt'))+
    geom_line(aes(y=data$xt,col='True Xt'))+
    geom_line(aes(y=data$estimation,col='Estimation'))+
    geom_point(aes(x=rep(1,T),y=data$particles,col='particle'))+
    geom_point(aes(x=rep(25,T),y=data$Particles25,col='particle at 25'))+
    geom_point(aes(x=rep(75,T),y=data$Particles75,col='particle at 75'))+  
    geom_point(aes(x=rep(T,T),y=data$initParticles,col='particle'))+
    ggtitle(paste('At SD = ',emission_sd))
}
ssm(emission_sd ,transition_sd )
```

Intermediate particles number 25 and 75 are chosen and their locations are mapped in the figure.

##B)Repeat the exercise above replacing the standard deviation of the emission model with 5 and then with 50. Comment on how this affects the results.

```{r 2, message=FALSE,error=FALSE,warning=FALSE}
emission_sd = 5
ssm(emission_sd ,transition_sd )

emission_sd = 50
ssm(emission_sd ,transition_sd )
```

Increasing the standard deviation not only makes the true x_t scattered but also scatters the particles both intermediate and endstep.

##C) Finally, show and explain what happens when the weights in the particle filter are always equal to 1, i.e. there is no correction.

```{r 3, message=FALSE,error=FALSE,warning=FALSE}
ssm <- function(emission_sd,transition_sd){
  T = 100
  particles = 100
  zt = xt = error = rep(NA,T)
  
  # for zt and xt
  zt[1] = initModel(1)
  for(i in 2:T){
    zt[i] <- transitionmodel(zt[i-1])
    xt[i] <- emmisionmodel(zt[i],emission_sd)
  }
  
  initParticles <- initModel(particles)
  particleWeights<- rep(1/particles,particles)
  estimation <- c()
  Particles25 = Particles75 = NA
  
  for(i in 2:T){
    newParticles <- sample(1:particles,particles,prob=particleWeights,replace = T)
    initParticles <- sapply(initParticles[newParticles],transitionmodel)
    
    if(i == 25){
      Particles25 <- c(initParticles)
    }
    else if(i == 75){
      Particles75 <- c(initParticles)
    }
    
    
    for(j in 1:particles){
      particleWeights[j] <- 1
    }
    #Normalise
    particleWeights <- particleWeights/sum(particleWeights)
    
    Ezt <- sum(particleWeights*initParticles)
    error[i] <- abs(zt[i]-Ezt)
    estimation[i] <- sum(particleWeights * initParticles)
  }
  
  data = data.frame(zt,xt,estimation,particles = initModel(particles),Particles25,Particles75,initParticles)
  
  ggplot(data,aes(x=c(1:T),y=value,color=variable,xlab="timestep"))+
    geom_line(aes(y=data$zt,col='True Zt'))+
    geom_line(aes(y=data$xt,col='True Xt'))+
    geom_line(aes(y=data$estimation,col='Estimation'))+
    geom_point(aes(x=rep(1,T),y=data$particles,col='particle'))+
    geom_point(aes(x=rep(25,T),y=data$Particles25,col='particle at 25'))+
    geom_point(aes(x=rep(75,T),y=data$Particles75,col='particle at 75'))+  
    geom_point(aes(x=rep(T,T),y=data$initParticles,col='particle'))+
    ggtitle(paste('At SD = ',emission_sd))
}

emission_sd = 1
ssm(emission_sd ,transition_sd )

emission_sd = 5
ssm(emission_sd ,transition_sd )

emission_sd = 50
ssm(emission_sd ,transition_sd )
```

It is seen that the estimation and true values do not match and accuracy is reduced. This change is consistent from all the standard deviation runs of 1,5 and 50.
value of the particle at all timesteps are scattered as compared to earlier runs. It reduces the quality of the filtering.



######LAB 4 stuff###
#install.packages("mvtnorm")
library("mvtnorm")

# Covariance function
SquaredExpKernel <- function(x1,x2,sigmaF=1,l=3){
  n1 <- length(x1)
  n2 <- length(x2)
  K <- matrix(NA,n1,n2)
  for (i in 1:n2){
    K[,i] <- sigmaF^2*exp(-0.5*( (x1-x2[i])/l)^2 )
  }
  return(K)
}

# Mean function
MeanFunc <- function(x){
  m <- sin(x)
  return(m)
}

# Simulates nSim realizations (function) from a GP with mean m(x) and covariance K(x,x')
# over a grid of inputs (x)
SimGP <- function(m = 0,K,x,nSim,...){
  n <- length(x)
  if (is.numeric(m)) meanVector <- rep(0,n) else meanVector <- m(x)
  covMat <- K(x,x,...)
  f <- rmvnorm(nSim, mean = meanVector, sigma = covMat)
  return(f)
}

xGrid <- seq(-5,5,length=20)

# Plotting one draw
sigmaF <- 1
l <- 1
nSim <- 1
fSim <- SimGP(m=MeanFunc, K=SquaredExpKernel, x=xGrid, nSim, sigmaF, l)
plot(xGrid, fSim[1,], type="p", ylim = c(-3,3))
if(nSim>1){
  for (i in 2:nSim) {
    lines(xGrid, fSim[i,], type="p")
  }
}
lines(xGrid,MeanFunc(xGrid), col = "red", lwd = 3)
lines(xGrid, MeanFunc(xGrid) - 1.96*sqrt(diag(SquaredExpKernel(xGrid,xGrid,sigmaF,l))), col = "blue", lwd = 2)
lines(xGrid, MeanFunc(xGrid) + 1.96*sqrt(diag(SquaredExpKernel(xGrid,xGrid,sigmaF,l))), col = "blue", lwd = 2)

# Plotting using manipulate package
library(manipulate)

plotGPPrior <- function(sigmaF, l, nSim){
  fSim <- SimGP(m=MeanFunc, K=SquaredExpKernel, x=xGrid, nSim, sigmaF, l)
  plot(xGrid, fSim[1,], type="l", ylim = c(-3,3), ylab="f(x)", xlab="x")
  if(nSim>1){
    for (i in 2:nSim) {
      lines(xGrid, fSim[i,], type="l")
    }
  }
  lines(xGrid,MeanFunc(xGrid), col = "red", lwd = 3)
  lines(xGrid, MeanFunc(xGrid) - 1.96*sqrt(diag(SquaredExpKernel(xGrid,xGrid,sigmaF,l))), col = "blue", lwd = 2)
  lines(xGrid, MeanFunc(xGrid) + 1.96*sqrt(diag(SquaredExpKernel(xGrid,xGrid,sigmaF,l))), col = "blue", lwd = 2)
  title(paste('length scale =',l,', sigmaf =',sigmaF))
}

manipulate(
  plotGPPrior(sigmaF, l, nSim = 10),
  sigmaF = slider(0, 2, step=0.1, initial = 1, label = "SigmaF"),
  l = slider(0, 2, step=0.1, initial = 1, label = "Length scale, l")
)

###Lab 4 setup#####
```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
library(kernlab)
library(AtmRay)
library(ggplot2)
tullinge  <- read.csv('https://github.com/STIMALiU/AdvMLCourse/raw/master/GaussianProcess/Code/TempTullinge.csv', header=TRUE, sep=';')
data <- read.csv('https://github.com/STIMALiU/AdvMLCourse/raw/master/GaussianProcess/Code/banknoteFraud.csv',
                 header=FALSE, sep=',')
```

# Assignment 1

## a)

```{r}
# The kernel function 
exp_kern <- function(x,xi,l, sigmaf ){
  
  return((sigmaf^2)*exp(-0.5*( (x - xi) / l )^2))
}
# The implementation, can take a custom kernel of any class.
linear_gp <- function(x,y,xStar,hyperParam,sigmaNoise,kernel){
  n <- length(x)  
  kernel_f <- kernel 
  # K = Covariance matrix calculation 
  K <- function(X, XI,...){ 
    
    kov <- matrix(0,nrow = length(X), ncol = length (XI)) 
    
    for(i in 1:length(XI)){
      
      kov[,i]<- kernel_f(X,XI[i],...) 
      
    }
    return(kov)
  }
  l <-hyperParam[1]
  sigmaf <- hyperParam[2]
  #K(X,X)
  K_xx   <- K(x,x, l = l, sigmaf = sigmaf) #, kernel = exp_kern
  #K(X*,X*) 
  K_xsxs <- K(xStar,xStar, l = l, sigmaf = sigmaf) # kernel = exp_kern,
  #K(X,X*)
  K_xxs  <- K(x,xStar, l = l, sigmaf = sigmaf)  #kernel = exp_kern,
  # Algorithm in page 19 of the Rasmus/Williams book
  sI <- sigmaNoise^2 * diag(dim(as.matrix(K_xx))[1])
  # L is transposed according to a definition in the R & W book
  L_transposed <- chol(K_xx + sI)
  L <- t(L_transposed)
  
  alpha <- solve(t(L), solve(L,y))
  f_bar_star <- t(K_xxs) %*% alpha
  v <- solve(L,K_xxs)
  V_fs <- K_xsxs - t(v) %*% v   
  log_mlike <- -0.5 %*% t(y) %*% alpha - sum( diag(L) - n/2 * log(2*pi) ) 
  return(list(fbar = f_bar_star, vf = V_fs, log_post= log_mlike))
}
```




```{r, echo = FALSE}
# Utility function for the tasks
plot_gp<- function(plot_it,band_it){
  ggplot() + 
    
    geom_point(
      aes(x = x, y = y), 
      data = plot_it,
      col = "blue",
      alpha = 0.7) + 
    
    geom_line(
      aes(x = xs, y = fbar),
      data = band_it, 
      alpha = 0.50) +
    
    geom_ribbon(
      aes(ymin = low, ymax = upp, xs), 
      data = band_it,
      alpha = 0.15) +
    
    theme_classic()
  
}
```




```{r}
# The data given
x <- c(-1.0, -0.6, -0.2, 0.4, 0.8)
y <- c(0.768, -0.044, -0.940, 0.719 ,-0.664)
# The noise
sn <- 0.1 
# The training grid
xs <- seq(-1,1,0.01)
# Hyperparameters l an sigma
hyperParam <- c(0.3, 1)
# Another utility function
repeter <- function(x,y,xs,sn,hyperParam,kernel){
  res <- linear_gp(x,y,xs,hyperParam,sn,kernel)
  # If you want the prediction band just add the noise variance (ie the sigma_n)
  upp <- res$fbar + 1.96*sqrt(diag(res$vf))
  low <- res$fbar - 1.96*sqrt(diag(res$vf))
  plot_it <- data.frame(x = x, y = y)
  band_it <- data.frame(xGrid = xs, fbar = res$fbar,  upp = upp, low = low)
  plot_gp(plot_it,band_it)
}
```

## b)

```{r}
repeter(x = x[4], y = y[4],xs,sn,hyperParam, kernel = exp_kern)
```

## c)

```{r}
repeter(x = x[c(2,4)], y = y[c(2,4)],xs,sn,hyperParam, kernel = exp_kern)
```

## d)

```{r}
repeter(x = x, y = y,xs,sn,hyperParam, kernel = exp_kern)
```

## e)

```{r}
x <- c(-1.0, -0.6, -0.2, 0.4, 0.8)
y <- c(0.768, -0.044, -0.940, 0.719 ,-0.664)
sn <- 0.1 
xs <- seq(-1,1,0.01)
hyperParam <- c(1, 1)
repeter(x = x[4], y = y[4],xs,sn,hyperParam, kernel = exp_kern)
repeter(x = x[c(2,4)], y = y[c(2,4)],xs,sn,hyperParam, kernel = exp_kern)
repeter(x = x, y = y,xs,sn,hyperParam, kernel = exp_kern)
```


# Assignment 2 

## Data preparations 

```{r}
tullinge$time <- 1:nrow(tullinge)
tullinge$day <- rep(1:365,6)
time_sub <- tullinge$time %in% seq(1,2190,5)
tullinge <- tullinge[time_sub,]
```


## a) 

```{r}
kern_maker <- function(l,sigmaf){
  
  exp_k <- function(x,y = NULL){
    
    return((sigmaf^2)*exp(-0.5*( (x - y) / l )^2))
  }
  
  class(exp_k) <- "kernel"
  return(exp_k)
}
```


```{r}
# gausspr()
# kernelMatrix()
ell <- 1
# SEkernel <- rbfdot(sigma = 1/(2*ell^2)) # Note how I reparametrize the rbfdo (which is the SE kernel) in kernlab
# SEkernel(1,2) 
my_exp <- kern_maker(l = 10, sigmaf =20)
x <- c(1,3,4)
x_star <- c(2,3,4)
#my_exp(x,x_star)
kernelMatrix(my_exp,x,x_star)
```


## b) 

```{r}
lm_tull <- lm(temp ~ time + I(time^2), data = tullinge)
sigma_2n <- var(resid(lm_tull))
a2b_kern <- kern_maker(l = 0.2, sigmaf = 20 )
gp_tullinge <- gausspr(x = tullinge$time, 
                       y = tullinge$temp, 
                       kernel = a2b_kern,
                       var = sigma_2n)
```

See task c) for the plot. 

## c) 

```{r}
sn_2c <- sqrt(sigma_2n)
xs_2c <- tullinge$time
hyperParam_2c <- c(0.2, 20)
res_2c<- linear_gp(x = tullinge$time, 
                   y = tullinge$temp,
                   xStar = xs_2c,
                   sigmaNoise  = sn_2c,
                   hyperParam = hyperParam_2c,
                   kernel = exp_kern)
upp2c <- predict(gp_tullinge) + 1.96*sqrt(diag(res_2c$vf))
low2c <- predict(gp_tullinge) - 1.96*sqrt(diag(res_2c$vf))
plot_it1 <- data.frame(x = tullinge$time, y = tullinge$temp)
band_it1 <- data.frame(xGrid = xs_2c, fbar = res_2c$fbar,  upp = upp2c, low = low2c)
```


```{r}
C2 <- ggplot() + 
  
  geom_point(
    aes(x = x, y = y), 
    data = plot_it1,
    col = "black",
    alpha = 0.7) + 
  
  geom_line(
    aes(x = xGrid, y = predict(gp_tullinge)),
    data = band_it1, 
    alpha = 1,
    col = "red") +
  
  geom_ribbon(
    aes(ymin = low2c, ymax = upp2c, x = xGrid), 
    data = band_it1,
    alpha = 0.2) + 
  
  theme_classic()
#plot(x=band_it1$xGrid, y=band_it1$fbar, type = "l")
C2
```

## d) 


```{r}
a2d_kern <- kern_maker(l = 1.2, sigmaf = 20 )
gp_tullinge_d <- gausspr(x = tullinge$day,
                         y = tullinge$temp, 
                         kernel = a2d_kern,
                         var = sigma_2n)
C23 <- C2 + geom_line(aes(x= tullinge$time,y = predict(gp_tullinge_d)), col = "blue", size = 0.8) 
#geom_point(data = tullinge, aes(x= time, y = temp)) + 


C23
# plot(y = tullinge$temp, x = tullinge$day)
# #lines(x = tullinge$time, y = fitted(lm_tull), col = "red")
# lines(x = tullinge$day, y = predict(gp_tullinge_d), col = "red" , lwd = 1)
```

The process model after time has an advantage in that sense that you can capture a trend isolated to a specific time since your modeling using the closest observations in time rather than the day model that assumes that the clostes related temperature point is the one on the same day previous years.

## e) 

```{r}
# periodic_kernel <- function(x,xi,sigmaf,d, l_1, l_2){
#  
#   part1 <- exp(2 * sin(pi * abs(x - xi) / d)^2 / l_1^2 )
#   part2 <- exp(-0.5 * abs(x - xi)^2 / l_2)
#   
#   sigmaf^2 * part1 * part2 
#   
# }
kern_maker2 <- function(sigmaf,d, l_1, l_2){
  
  periodic_kernel <- function(x,y = NULL){
    
    part1 <- exp(-2 * sin(pi * abs(x - y) / d)^2 / l_1^2 )
    part2 <- exp(-0.5 * abs(x - y)^2 / l_2^2)
    
    sigmaf^2 * part1 * part2 
    
  }
  
  class(periodic_kernel) <- "kernel"
  return(periodic_kernel)
}
```


```{r}
sigmaff <- 20 
l1 <- 1 
l2 <- 10  
d_est <- 365 / sd(tullinge$time)

periodic_kernel <- kern_maker2(sigmaf = sigmaff,
                               d = d_est , 
                               l_1 = l1, 
                               l_2 = l2)
gp_tullinge_et <- gausspr(x = tullinge$time, 
                          y = tullinge$temp, 
                          kernel = periodic_kernel,
                          var = sigma_2n)
gp_tullinge_ed <- gausspr(x = tullinge$day, 
                          y = tullinge$temp, 
                          kernel = periodic_kernel,
                          var = sigma_2n)
```


```{r}
ggplot(data = tullinge, aes(x= time, y = temp)) +
  geom_point() + 
  geom_line(aes(y = predict(gp_tullinge_et)), col = "darkgreen", size = 0.8) 
```

This kernel seems to catch both variation in day and over time.

# Assignment 3

```{r}
names(data) <- c("varWave","skewWave","kurtWave","entropyWave","fraud")
data[,5] <- as.factor(data[,5])
set.seed(111) 
SelectTraining <- sample(1:dim(data)[1], size = 1000, replace = FALSE)
train <- data[SelectTraining,]
test <- data[-SelectTraining,]
```


## a) 

```{r}
colnames(data)
GPfitFraud <- gausspr(fraud ~  varWave + skewWave, data = train)
GPfitFraud
# predict on the test set
fit_train<- predict(GPfitFraud,train[,c("varWave","skewWave")])
table(fit_train, train$fraud) # confusion matrix
mean(fit_train == train$fraud)
```


```{r}
# probPreds <- predict(GPfitIris, iris[,3:4], type="probabilities")
x1 <- seq(min(data[,"varWave"]),max(data[,"varWave"]),length=100)
x2 <- seq(min(data[,"skewWave"]),max(data[,"skewWave"]),length=100)
gridPoints <- meshgrid(x1, x2)
gridPoints <- cbind(c(gridPoints$x), c(gridPoints$y))
gridPoints <- data.frame(gridPoints)
names(gridPoints) <- c("varWave","skewWave")
probPreds <- predict(GPfitFraud, gridPoints, type="probabilities")
contour(x1,x2,t(matrix(probPreds[,1],100)), 20, 
        xlab = "varWave", ylab = "skewWave", 
        main = 'Prob(Fraud) - Fraud is red')
points(data[data[,5]== 1,"varWave"],data[data[,5]== 1,"skewWave"],col="blue")
points(data[data[,5]== 0,"varWave"],data[data[,5]== 0,"skewWave"],col="red")
```


## b) 


```{r}
# predict on the test set
fit_test<- predict(GPfitFraud,test[,c("varWave","skewWave")])
table(fit_test, test$fraud) # confusion matrix
mean(fit_test == test$fraud)
```

## c) 

```{r}
GPfitFraudFull <- gausspr(fraud ~ ., data = train)
GPfitFraudFull
# predict on the test set
fit_Full<- predict(GPfitFraudFull,test[,-ncol(test)])
table(fit_Full, test$fraud) # confusion matrix
mean(fit_Full == test$fraud)
```


