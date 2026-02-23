import asyncio
import pytafast

async def SMA(*args, **kwargs):
    return await asyncio.to_thread(pytafast.SMA, *args, **kwargs)

async def EMA(*args, **kwargs):
    return await asyncio.to_thread(pytafast.EMA, *args, **kwargs)

async def RSI(*args, **kwargs):
    return await asyncio.to_thread(pytafast.RSI, *args, **kwargs)

async def MACD(*args, **kwargs):
    return await asyncio.to_thread(pytafast.MACD, *args, **kwargs)

async def BBANDS(*args, **kwargs):
    return await asyncio.to_thread(pytafast.BBANDS, *args, **kwargs)

async def ATR(*args, **kwargs):
    return await asyncio.to_thread(pytafast.ATR, *args, **kwargs)

async def ADX(*args, **kwargs):
    return await asyncio.to_thread(pytafast.ADX, *args, **kwargs)

async def CCI(*args, **kwargs):
    return await asyncio.to_thread(pytafast.CCI, *args, **kwargs)

async def OBV(*args, **kwargs):
    return await asyncio.to_thread(pytafast.OBV, *args, **kwargs)

async def ROC(*args, **kwargs):
    return await asyncio.to_thread(pytafast.ROC, *args, **kwargs)

async def STOCH(*args, **kwargs):
    return await asyncio.to_thread(pytafast.STOCH, *args, **kwargs)
