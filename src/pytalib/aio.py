import asyncio
import pytalib

async def SMA(*args, **kwargs):
    return await asyncio.to_thread(pytalib.SMA, *args, **kwargs)

async def EMA(*args, **kwargs):
    return await asyncio.to_thread(pytalib.EMA, *args, **kwargs)

async def RSI(*args, **kwargs):
    return await asyncio.to_thread(pytalib.RSI, *args, **kwargs)

async def MACD(*args, **kwargs):
    return await asyncio.to_thread(pytalib.MACD, *args, **kwargs)

async def BBANDS(*args, **kwargs):
    return await asyncio.to_thread(pytalib.BBANDS, *args, **kwargs)

async def ATR(*args, **kwargs):
    return await asyncio.to_thread(pytalib.ATR, *args, **kwargs)

async def ADX(*args, **kwargs):
    return await asyncio.to_thread(pytalib.ADX, *args, **kwargs)

async def CCI(*args, **kwargs):
    return await asyncio.to_thread(pytalib.CCI, *args, **kwargs)

async def OBV(*args, **kwargs):
    return await asyncio.to_thread(pytalib.OBV, *args, **kwargs)

async def ROC(*args, **kwargs):
    return await asyncio.to_thread(pytalib.ROC, *args, **kwargs)

async def STOCH(*args, **kwargs):
    return await asyncio.to_thread(pytalib.STOCH, *args, **kwargs)
