library(quantmod)
library(TTR)

# Load data
df <- read.csv("data/nasdaq100_2025_now.csv")
df$Date <- as.Date(df$Date)
xts_data <- xts(df[, c("Open", "High", "Low", "Close", "Volume")], order.by = df$Date)

# Setup layout
png("r_chart.png", width = 1200, height = 1000, res = 100)

# Main chart with Candlestick
chartSeries(xts_data, theme = chartTheme("white"), TA = NULL, name = "NASDAQ 100 (R)")

# Add Ichimoku
n1 <- 9; n2 <- 26; n3 <- 52
H <- Hi(xts_data); L <- Lo(xts_data)
tenkan <- (runMax(H, n1) + runMin(L, n1)) / 2
kijun <- (runMax(H, n2) + runMin(L, n2)) / 2
ssa <- (tenkan + kijun) / 2
ssb <- (runMax(H, n3) + runMin(L, n3)) / 2

addTA(tenkan, on = 1, col = "blue", lwd = 1, name = "Tenkan")
addTA(kijun, on = 1, col = "red", lwd = 1, name = "Kijun")
# In base R, we cannot easily shift and shade future spans, but we can plot overlays.
addTA(ssa, on = 1, col = "green", lwd = 1, name = "SenkouA")
addTA(ssb, on = 1, col = "brown", lwd = 1, name = "SenkouB")

# Add ADX
addADX(n = 14)

# Add Momentum
addMomentum(n = 10)

dev.off()
print("R chart saved to r_chart.png")
