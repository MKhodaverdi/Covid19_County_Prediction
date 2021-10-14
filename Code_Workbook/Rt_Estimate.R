# calculating Rt 

library(dplyr)
library(scales)
library(magrittr)
library(zoo)
library(EpiEstim)

# getting data ----------------------------------------------------------------
workDir="C:/Users/khodaverdi/Desktop/R_code/"
setwd(workDir)

data <- read.csv("test_data.csv")
si<-read.csv("serial_interval.csv")

dates <- as.Date(unique(data$dateRep), format='%Y-%m-%d')
counties <- unique(data$county)


# serial interval ------------------------------------------------------
si<-si$fit[1:40]
si<-c(0,si)
tot<-sum(si)
si<-si/tot


# R_tables -------------------------------------------------------------
RTable<-list()

for (county in counties)
{
  data_sub <- data[data$county==county,]
  DC <-list()
  DC$dates <- dates[1:length(dates)] 
  DC$I <- data_sub$casesTot[1:length(dates)]  
  DC <- as.data.frame(DC)
  
  if (sum(DC$I)>0){
    
    R_BW <- estimate_R(incid=DC, method = "non_parametric_si",
                        config = list(t_start = seq(2, nrow(DC)-13),
                                      t_end = seq(15, nrow(DC)),
                                      si_distr = si ))
    R_W<-estimate_R(incid=DC,method="non_parametric_si",
                    config = make_config(si_distr=si))  
    
    
    RTable$county <- c(RTable$county, rep(county,nrow(R_BW$R)))
    
    # for LSTM with 14 days window
    RTable$t_endT <- c(RTable$t_endT, R_BW$R[,2])
    RTable$meanT <- c(RTable$meanT, R_BW$R[,3])
    RTable$stdT <- c(RTable$std, R_BW$R[,4])    
    RTable$Q5T <- c(RTable$Q5T, R_BW$R[,6])
    RTable$Q95T <- c(RTable$Q95T, R_BW$R[,10])
    
    # for LSTM with 7 day window
    RTable$t_startK <-c (RTable$t_endK, R_W$R[8:nrow(R_W$R),2])
    RTable$meanK <- c(RTable$meanK, R_W$R[8:nrow(R_W$R),3])
    RTable$stdK <- c(RTable$stdK, R_W$R[8:nrow(R_W$R),4])   
    RTable$Q5K <- c(RTable$Q5K, R_W$R[8:nrow(R_W$R),6])
    RTable$Q95K <- c(RTable$Q95K, R_W$R[8:nrow(R_W$R),10])
    
    alpha <- (RTable$meanT/RTable$stdT)^2
    theta <- (RTable$stdT^2)/RTable$meanT
    RTable$probT <- 1-pgamma(q =1 , shape = alpha, scale = theta)  # p(R>1)
    alpha <- (RTable$meanK/RTable$stdK)^2
    theta <- (RTable$stdK^2)/RTable$meanK
    RTable$probK <- 1-pgamma(q =1 , shape = alpha, scale = theta)
    
    RTable$incid <- c(RTable$incid, DC$I[15:length(dates)])
  }
}


# Final numbers ------------------------------------------------------

RTable2 <- as.data.frame(RTable) %>%
  left_join( data.frame("t_endT"=seq(1:length(dates)), "date"=dates[1:length(dates)]) ) %>%
  select(15,1,14,3,4,5,6,12,8,9,10,11,13) %>% 
  rename(R_exp7=meanT, R_sig7=stdT, R_param_a7=Q5T , R_param_b7=Q95T, Prob_R7=probT, 
         R_exp14=meanK, R_sig14=stdK, R_param_a14=Q5K , R_param_b14=Q95K, Prob_R14=probK)

RTable3 <- as.data.frame(RTable) %>%
  left_join( data.frame("t_endT"=seq(1:length(dates)), "date"=dates[1:length(dates)]) ) %>%
  select(15,1,14,3,4,5,6,12,8,9,10,11,13) %>% 
  rename(R_exp7=meanT, R_sig7=stdT, R_param_a7=Q5T , R_param_b7=Q95T, Prob_R7=probT, 
         R_exp14=meanK, R_sig14=stdK, R_param_a14=Q5K , R_param_b14=Q95K, Prob_R14=probK)

write.csv(format(RTable2, digits=2, scipen=999), file="RTable_LSTM_tot.csv", row.names=FALSE, quote=FALSE)  
write.csv(format(RTable3, digits=2, scipen=999), file="RTable_LSTM_tot1.csv", row.names=FALSE, quote=FALSE) 

