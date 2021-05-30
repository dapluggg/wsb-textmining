#install.packages("quantmod")
library(quantmod)
start <- as.Date("2020-11-14")
end <- as.Date("2021-05-14")
getSymbols("GME", src = "yahoo", from = start, to = end)
#head(GME)

#Grab Price
price.GME <- quantmod::Cl(GME)

#RSI lag time 
day <-10

# Calculate RSI
GME.RSI <- round(TTR::RSI(price.GME, day),2)
rsidata <- as.data.frame(GME.RSI)
names(rsidata) <- "GME RSI"

#Combine datasets 
retcomb.GME <- cbind(rsidata, GME)
cleanGME <- retcomb.GME[-c(1:day),]
#cleanGME

#AMC
getSymbols("AMC", src = "yahoo", from = start, to = end)

#Grab Price
price.AMC <- quantmod::Cl(AMC)

# Calculate RSI
AMC.RSI <- round(TTR::RSI(price.AMC, day),2)
rsidata.AMC <- as.data.frame(AMC.RSI)
names(rsidata.AMC) <- "AMC RSI"

#Combine datasets 
retcomb.AMC <- cbind(rsidata.AMC, AMC)
cleanAMC <- retcomb.AMC[-c(1:day),]

#Combine datasets 
cleandata <- cbind(cleanGME,cleanAMC)
df <- cbind(Date = rownames(cleandata), cleandata)
rownames(df) <- 1:nrow(df)
head(df)

write.csv(df, "C:/Users/KnudseQ/Desktop/cleandata.csv",row.names = FALSE)
