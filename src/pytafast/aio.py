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

async def MOM(*args, **kwargs):
    return await asyncio.to_thread(pytafast.MOM, *args, **kwargs)

async def STDDEV(*args, **kwargs):
    return await asyncio.to_thread(pytafast.STDDEV, *args, **kwargs)

async def WILLR(*args, **kwargs):
    return await asyncio.to_thread(pytafast.WILLR, *args, **kwargs)

async def NATR(*args, **kwargs):
    return await asyncio.to_thread(pytafast.NATR, *args, **kwargs)

async def MFI(*args, **kwargs):
    return await asyncio.to_thread(pytafast.MFI, *args, **kwargs)

async def CMO(*args, **kwargs):
    return await asyncio.to_thread(pytafast.CMO, *args, **kwargs)

async def DX(*args, **kwargs):
    return await asyncio.to_thread(pytafast.DX, *args, **kwargs)

async def MINUS_DI(*args, **kwargs):
    return await asyncio.to_thread(pytafast.MINUS_DI, *args, **kwargs)

async def MINUS_DM(*args, **kwargs):
    return await asyncio.to_thread(pytafast.MINUS_DM, *args, **kwargs)

async def PLUS_DI(*args, **kwargs):
    return await asyncio.to_thread(pytafast.PLUS_DI, *args, **kwargs)

async def PLUS_DM(*args, **kwargs):
    return await asyncio.to_thread(pytafast.PLUS_DM, *args, **kwargs)

async def APO(*args, **kwargs):
    return await asyncio.to_thread(pytafast.APO, *args, **kwargs)

async def AROON(*args, **kwargs):
    return await asyncio.to_thread(pytafast.AROON, *args, **kwargs)

async def AROONOSC(*args, **kwargs):
    return await asyncio.to_thread(pytafast.AROONOSC, *args, **kwargs)

async def PPO(*args, **kwargs):
    return await asyncio.to_thread(pytafast.PPO, *args, **kwargs)

async def TRIX(*args, **kwargs):
    return await asyncio.to_thread(pytafast.TRIX, *args, **kwargs)

async def ULTOSC(*args, **kwargs):
    return await asyncio.to_thread(pytafast.ULTOSC, *args, **kwargs)
