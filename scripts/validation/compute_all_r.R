library(TTR)

# Load data
data_file <- Sys.getenv("DATA_FILE", unset = "data/berkshire_1y.csv")
data <- read.csv(data_file)
data$Date <- as.Date(data$Date)
data <- data[order(data$Date), ]

O <- data$Open
H <- data$High
L <- data$Low
C <- data$Close
V <- as.numeric(data$Volume)

results <- data.frame(Date = data$Date)

# --- Overlap ---
results$SMA <- as.numeric(SMA(C, 30))
results$EMA <- as.numeric(EMA(C, 30))
results$WMA <- as.numeric(WMA(C, 30))
results$SAR <- as.numeric(SAR(cbind(H, L)))
bb <- BBands(C, n=5, sd=2, maType="SMA")
results$BBANDS_0 <- as.numeric(bb[, "up"])
results$BBANDS_1 <- as.numeric(bb[, "mavg"])
results$BBANDS_2 <- as.numeric(bb[, "dn"])

# --- Momentum ---
results$RSI <- as.numeric(RSI(C, n = 14, maType = "EMA", wilder = TRUE))
macd_res <- MACD(C, nFast = 12, nSlow = 26, nSig = 9, maType = "EMA", percent = FALSE)
results$MACD_0 <- as.numeric(macd_res[, "macd"])
results$MACD_1 <- as.numeric(macd_res[, "signal"])

results$MOM <- as.numeric(momentum(C, n = 10))
results$ROC <- as.numeric(ROC(C, n = 10, type = "discrete") * 100)
results$ROCP <- as.numeric(ROC(C, n = 10, type = "discrete"))

# Manual Lag for ROCR
c_lag10 <- c(rep(NA, 10), head(C, -10))
results$ROCR <- as.numeric(C / c_lag10)
results$ROCR100 <- as.numeric((C / c_lag10) * 100)

results$CCI <- as.numeric(CCI(cbind(H, L, C), n = 14))
results$MFI <- as.numeric(MFI(cbind(H, L, C), V, n = 14))
wpr <- WPR(cbind(H, L, C), n = 14)
results$WILLR <- as.numeric((wpr - 1) * 100)

aroon_res <- aroon(cbind(H, L), n = 14)
results$AROON_0 <- as.numeric(aroon_res[, "aroonDn"])
results$AROON_1 <- as.numeric(aroon_res[, "aroonUp"])
results$AROONOSC <- as.numeric(results$AROON_1 - results$AROON_0)

# --- Volatility ---
atr_res <- ATR(cbind(H, L, C), n = 14, maType = "EMA", wilder = TRUE)
results$ATR <- as.numeric(atr_res[, "atr"])
results$TRANGE <- as.numeric(TR(cbind(H, L, C))[, "tr"])
results$STDDEV <- as.numeric(runSD(C, n = 5))

# --- Volume ---
results$OBV <- as.numeric(OBV(C, V))
results$AD <- as.numeric(chaikinAD(cbind(H, L, C), V))

# --- Price ---
results$TYPPRICE <- (H + L + C) / 3
results$WCLPRICE <- (H + L + C*2) / 4
results$MEDPRICE <- (H + L) / 2
results$AVGPRICE <- (O + H + L + C) / 4

# --- Math ---
results$ADD <- H + L
results$SUB <- H - L
results$MULT <- H * L
results$DIV <- H / L
results$SQRT <- sqrt(C)
results$LN <- log(C)
results$LOG10 <- log10(C)
results$SIN <- sin(C)
results$COS <- cos(C)
results$TAN <- tan(C)

# --- R-consistent Indicators ---
# MAs
results$ZLEMA <- as.numeric(ZLEMA(C, n = 30))
results$HMA <- as.numeric(HMA(C))
results$ALMA <- as.numeric(ALMA(C))
results$EVWMA <- as.numeric(EVWMA(C, V, n = 30))

# Channels
kc <- keltnerChannels(cbind(H, L, C))
results$Keltner_up <- as.numeric(kc[, "up"])
results$Keltner_mid <- as.numeric(kc[, "mavg"])
results$Keltner_dn <- as.numeric(kc[, "dn"])

# Oscillators
results$CMF <- as.numeric(CMF(cbind(H, L, C), V))
results$DPO <- as.numeric(DPO(C))
results$CMO <- as.numeric(CMO(C, n=14))
emv_r <- EMV(cbind(H, L), V)
results$EMV_emv <- as.numeric(emv_r[, "emv"])
results$EMV_ma <- as.numeric(emv_r[, "maEMV"])
smi_r <- SMI(cbind(H, L, C))
results$SMI_smi <- as.numeric(smi_r[, "SMI"])
results$SMI_signal <- as.numeric(smi_r[, "signal"])

# CLV and CHV
results$CLV <- as.numeric(CLV(cbind(H, L, C)))
results$CHV <- as.numeric(chaikinVolatility(cbind(H, L), n=10) * 100)

# Special
results$VHF <- as.numeric(VHF(C))
results$SNR <- as.numeric(SNR(cbind(H, L, C), n = 14))

# Legacy ones
dc <- DonchianChannel(cbind(H, L), n = 10)
results$Donchian_high <- as.numeric(dc[, "high"])
results$Donchian_mid <- as.numeric(dc[, "mid"])
results$Donchian_low <- as.numeric(dc[, "low"])
results$ZigZag <- as.numeric(ZigZag(cbind(H, L), change = 5.0, percent = TRUE))
gmma_r <- GMMA(C)
for (i in 1:12) {
  results[[paste0("GMMA_", i-1)]] <- as.numeric(gmma_r[, i])
}
kst_r <- KST(C)
results$KST_kst <- as.numeric(kst_r[, "kst"])
results$KST_signal <- as.numeric(kst_r[, "signal"])

# --- New Indicators ---
# ADX and DI
adx_res <- ADX(cbind(H, L, C), n = 14)
results$ADX <- as.numeric(adx_res[, "ADX"])
results$PLUS_DI <- as.numeric(adx_res[, "DIp"])
results$MINUS_DI <- as.numeric(adx_res[, "DIn"])

# Ichimoku
n1 <- 9; n2 <- 26; n3 <- 52
results$Tenkan <- (runMax(H, n1) + runMin(L, n1)) / 2
results$Kijun <- (runMax(H, n2) + runMin(L, n2)) / 2
# Spans are shifted forward in plotting, so we just check the base values here
results$SenkouA <- (results$Tenkan + results$Kijun) / 2
results$SenkouB <- (runMax(H, n3) + runMin(L, n3)) / 2

# TDI (Traders Dynamic Index)
# RSI(13), Price Line = SMA(RSI, 2), Signal Line = SMA(RSI, 7)
# BB(RSI, 34, 1.6185), Market Base Line = SMA(RSI, 34)
tdi_rsi <- as.numeric(RSI(C, n = 13, maType = "EMA", wilder = TRUE))
results$TDI_price <- as.numeric(SMA(tdi_rsi, 2))
results$TDI_signal <- as.numeric(SMA(tdi_rsi, 7))
tdi_bb <- BBands(tdi_rsi, n = 34, sd = 1.6185, maType = "SMA")
results$TDI_mbl <- as.numeric(tdi_bb[, "mavg"])
results$TDI_ub <- as.numeric(tdi_bb[, "up"])
results$TDI_lb <- as.numeric(tdi_bb[, "dn"])

# Output configuration
output_file <- Sys.getenv("OUTPUT_FILE", unset = "data/r_all_results.csv")

write.csv(results, output_file, row.names = FALSE)
print(paste("R results exported to", output_file))
