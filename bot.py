import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import io
import requests
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, ContextTypes

# Configurar logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configuración de Telegram
TELEGRAM_TOKEN = "TU_TELEGRAM_BOT_TOKEN"  # Reemplaza con tu token
TELEGRAM_CHAT_ID = "TU_CHAT_ID"  # Reemplaza con tu chat ID

class TelegramNotifier:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self.bot = Bot(token=token)
    
    async def send_message(self, text):
        """Envía un mensaje de texto a Telegram"""
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=text)
        except Exception as e:
            print(f"Error enviando mensaje a Telegram: {e}")
    
    async def send_photo(self, image_buffer, caption=""):
        """Envía una imagen a Telegram"""
        try:
            image_buffer.seek(0)
            await self.bot.send_photo(chat_id=self.chat_id, photo=image_buffer, caption=caption)
        except Exception as e:
            print(f"Error enviando imagen a Telegram: {e}")

class PatternAnalyzer:
    def __init__(self):
        pass
    
    def get_ohlcv_data(self, symbol, timeframe, period="5d"):
        try:
            # Obtener datos de Yahoo Finance
            ticker = yf.Ticker(symbol)
            df = ticker.history(interval=timeframe, period=period)
            
            if df.empty:
                logger.error(f"No se pudieron obtener datos para {symbol}")
                return None
                
            return df
        except Exception as e:
            logger.error(f"Error obteniendo datos para {symbol} en {timeframe}: {e}")
            return None
    
    def detect_patterns(self, df, symbol, timeframe):
        patterns = []
        
        # Verificar que tenemos datos suficientes
        if df is None or len(df) < 20:
            return patterns
        
        # Detectar patrones personalizados
        if self.detect_double_top(df):
            patterns.append("Doble Techo (bajista)")
        
        if self.detect_double_bottom(df):
            patterns.append("Doble Suelo (alcista)")
            
        if self.detect_head_shoulders(df):
            patterns.append("Cabeza y Hombros (bajista)")
            
        if self.detect_inverse_head_shoulders(df):
            patterns.append("Cabeza y Hombros Invertido (alcista)")
            
        if self.detect_triangle(df):
            patterns.append("Triángulo Simétrico")
            
        if self.detect_rising_wedge(df):
            patterns.append("Cuña Ascendente (bajista)")
            
        if self.detect_falling_wedge(df):
            patterns.append("Cuña Descendente (alcista)")
            
        if self.detect_bullish_flag(df):
            patterns.append("Bandera Alcista")
            
        if self.detect_bearish_flag(df):
            patterns.append("Bandera Bajista")
        
        return patterns
    
    # [Todas las funciones de detección de patrones permanecen igual que antes]
    # ... (omitiendo por brevedad, pero deben estar presentes en tu código)

class Backtester:
    def __init__(self, telegram_notifier=None):
        self.analyzer = PatternAnalyzer()
        self.results = []
        self.notifier = telegram_notifier
    
    def download_historical_data(self, symbol, timeframe, days=60):
        """Descarga datos históricos de Yahoo Finance con el período adecuado"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Determinar el período basado en el timeframe
            if timeframe == "15m":
                period = "60d"  # Máximo permitido por Yahoo Finance para 15m
            elif timeframe == "1h":
                period = "730d"  # Máximo permitido para 1h
            else:
                period = f"{days}d"
            
            df = ticker.history(interval=timeframe, period=period)
            return df
        except Exception as e:
            print(f"Error descargando datos para {symbol}: {e}")
            return None
    
    async def run_backtest(self, symbol, timeframe, days=60, initial_balance=10000):
        """Ejecuta backtesting para un símbolo y timeframe específico"""
        message = f"🔍 Iniciando backtest para {symbol} ({timeframe}) para los últimos {days} días"
        if self.notifier:
            await self.notifier.send_message(message)
        else:
            print(message)
        
        # Descargar datos históricos
        df = self.download_historical_data(symbol, timeframe, days)
        if df is None or df.empty:
            error_msg = f"No se pudieron obtener datos para {symbol} en {timeframe}"
            if self.notifier:
                await self.notifier.send_message(error_msg)
            else:
                print(error_msg)
            return None
        
        # Preparar variables para el backtest
        balance = initial_balance
        position = 0  # 0: sin posición, 1: larga, -1: corta
        entry_price = 0
        trades = []
        
        # Iterar através de los datos
        for i in range(20, len(df)):
            current_data = df.iloc[:i]  # Datos hasta el punto actual
            
            # Detectar patrones
            patterns = self.analyzer.detect_patterns(current_data, symbol, timeframe)
            
            # Obtener precio actual
            current_price = current_data['Close'].iloc[-1]
            current_date = current_data.index[-1]
            
            # Estrategia de trading basada en patrones
            if patterns and position == 0:  # Si detectamos patrones y no tenemos posición
                # Estrategia simple: comprar en patrones alcistas, vender en patrones bajistas
                bullish_patterns = [p for p in patterns if 'alcista' in p.lower() or 'suelo' in p.lower()]
                bearish_patterns = [p for p in patterns if 'bajista' in p.lower() or 'techo' in p.lower()]
                
                if bullish_patterns:
                    # Entrar en posición larga
                    position = 1
                    entry_price = current_price
                    trades.append({
                        'date': current_date,
                        'action': 'BUY',
                        'price': current_price,
                        'balance': balance,
                        'patterns': bullish_patterns
                    })
                    trade_msg = f"{current_date}: COMPRA a {current_price} por patrones {bullish_patterns}"
                    if self.notifier:
                        await self.notifier.send_message(trade_msg)
                    else:
                        print(trade_msg)
                
                elif bearish_patterns:
                    # Entrar en posición corta
                    position = -1
                    entry_price = current_price
                    trades.append({
                        'date': current_date,
                        'action': 'SELL',
                        'price': current_price,
                        'balance': balance,
                        'patterns': bearish_patterns
                    })
                    trade_msg = f"{current_date}: VENTA a {current_price} por patrones {bearish_patterns}"
                    if self.notifier:
                        await self.notifier.send_message(trade_msg)
                    else:
                        print(trade_msg)
            
            # Salir de la posición (estrategia simple: salir después de 5 velas o con 5% de ganancia/pérdida)
            elif position != 0:
                holding_period = 5  # Salir después de 5 velas
                profit_target = 0.05  # 5% de objetivo de ganancia
                stop_loss = 0.03  # 3% de stop loss
                
                # Calcular ganancia/pérdida
                if position == 1:  # Posición larga
                    profit_pct = (current_price - entry_price) / entry_price
                    # Convertir timedelta a horas para timeframe de 1h
                    if timeframe == "1h":
                        hours_held = (current_date - trades[-1]['date']).total_seconds() / 3600
                        exit_condition = (profit_pct >= profit_target or 
                                         profit_pct <= -stop_loss or 
                                         hours_held >= holding_period)
                    else:  # Para 15m, usar número de velas
                        candles_held = len(df) - i
                        exit_condition = (profit_pct >= profit_target or 
                                         profit_pct <= -stop_loss or 
                                         candles_held >= holding_period)
                
                elif position == -1:  # Posición corta
                    profit_pct = (entry_price - current_price) / entry_price
                    if timeframe == "1h":
                        hours_held = (current_date - trades[-1]['date']).total_seconds() / 3600
                        exit_condition = (profit_pct >= profit_target or 
                                         profit_pct <= -stop_loss or 
                                         hours_held >= holding_period)
                    else:
                        candles_held = len(df) - i
                        exit_condition = (profit_pct >= profit_target or 
                                         profit_pct <= -stop_loss or 
                                         candles_held >= holding_period)
                
                if exit_condition:
                    # Salir de la posición
                    if position == 1:
                        balance *= (1 + profit_pct)
                        action = 'SELL'
                    else:
                        balance *= (1 + profit_pct)
                        action = 'BUY_TO_COVER'
                    
                    trades.append({
                        'date': current_date,
                        'action': action,
                        'price': current_price,
                        'balance': balance,
                        'profit_pct': profit_pct
                    })
                    trade_msg = f"{current_date}: {action} a {current_price} - Balance: {balance:.2f} ({profit_pct*100:.2f}%)"
                    if self.notifier:
                        await self.notifier.send_message(trade_msg)
                    else:
                        print(trade_msg)
                    position = 0
        
        # Calcular métricas de performance
        final_balance = balance
        total_return = (final_balance - initial_balance) / initial_balance * 100
        num_trades = len([t for t in trades if t['action'] in ['BUY', 'SELL']])
        winning_trades = len([t for t in trades if 'profit_pct' in t and t['profit_pct'] > 0])
        win_rate = winning_trades / num_trades * 100 if num_trades > 0 else 0
        
        result = {
            'symbol': symbol,
            'timeframe': timeframe,
            'initial_balance': initial_balance,
            'final_balance': final_balance,
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'trades': trades
        }
        
        self.results.append(result)
        return result
    
    async def run_multiple_backtests(self, symbols, timeframes, days=60, initial_balance=10000):
        """Ejecuta backtests para múltiples símbolos y timeframes"""
        all_results = []
        
        for symbol in symbols:
            for timeframe in timeframes:
                result = await self.run_backtest(symbol, timeframe, days, initial_balance)
                if result:
                    all_results.append(result)
        
        return all_results
    
    async def generate_report(self, results):
        """Genera un reporte de los resultados del backtesting y lo envía a Telegram"""
        if not results:
            message = "No hay resultados para reportar"
            if self.notifier:
                await self.notifier.send_message(message)
            else:
                print(message)
            return
        
        # Crear mensaje de reporte
        report_message = "📊 REPORTE DE BACKTESTING - PATRONES CHARTISTAS\n\n"
        
        for result in results:
            report_message += f"✅ {result['symbol']} ({result['timeframe']})\n"
            report_message += f"   Balance inicial: ${result['initial_balance']:.2f}\n"
            report_message += f"   Balance final: ${result['final_balance']:.2f}\n"
            report_message += f"   Retorno total: {result['total_return']:.2f}%\n"
            report_message += f"   Operaciones: {result['num_trades']}\n"
            report_message += f"   Ratio de aciertos: {result['win_rate']:.2f}%\n\n"
        
        # Calcular promedios
        if results:
            avg_return = np.mean([r['total_return'] for r in results])
            avg_win_rate = np.mean([r['win_rate'] for r in results])
            
            report_message += f"📈 RESUMEN GENERAL:\n"
            report_message += f"   Retorno promedio: {avg_return:.2f}%\n"
            report_message += f"   Ratio de acierto promedio: {avg_win_rate:.2f}%\n"
        
        # Enviar reporte a Telegram
        if self.notifier:
            await self.notifier.send_message(report_message)
        else:
            print(report_message)
        
        return report_message
    
    async def plot_results(self, results):
        """Genera gráficos de los resultados y los envía a Telegram"""
        if not results:
            return
        
        # Gráfico de balances
        plt.figure(figsize=(12, 6))
        for result in results:
            # Extraer balances a lo largo del tiempo
            balances = [result['initial_balance']]
            dates = [result['trades'][0]['date'] if result['trades'] else datetime.now()]
            
            for trade in result['trades']:
                if 'balance' in trade:
                    balances.append(trade['balance'])
                    dates.append(trade['date'])
            
            if len(dates) > 1:
                label = f"{result['symbol']} ({result['timeframe']}) - {result['total_return']:.2f}%"
                plt.plot(dates, balances, label=label, marker='o', markersize=3)
        
        plt.title('Evolución del Balance')
        plt.xlabel('Fecha')
        plt.ylabel('Balance ($)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Guardar gráfico en buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        # Enviar gráfico a Telegram
        if self.notifier:
            await self.notifier.send_photo(buf, caption="Evolución del Balance en Backtesting")
        
        # Gráfico de distribución de retornos (solo si hay múltiples resultados)
        if len(results) > 1:
            returns = [r['total_return'] for r in results]
            plt.figure(figsize=(10, 6))
            plt.hist(returns, bins=20, alpha=0.7, edgecolor='black')
            plt.axvline(np.mean(returns), color='red', linestyle='dashed', linewidth=1, label=f'Media: {np.mean(returns):.2f}%')
            plt.title('Distribución de Retornos')
            plt.xlabel('Retorno (%)')
            plt.ylabel('Frecuencia')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            # Guardar gráfico en buffer
            buf2 = io.BytesIO()
            plt.savefig(buf2, format='png')
            buf2.seek(0)
            plt.close()
            
            # Enviar gráfico a Telegram
            if self.notifier:
                await self.notifier.send_photo(buf2, caption="Distribución de Retornos")

# Función principal para ejecutar el backtesting y enviar resultados a Telegram
async def main():
    # Configuración para datos intraday (15m, 1h)
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
    timeframes = ["15m", "1h"]
    days = 30  # Últimos 30 días (dentro del límite de Yahoo Finance)
    initial_balance = 10000
    
    # Inicializar notificador de Telegram
    notifier = TelegramNotifier(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
    
    # Crear backtester con notificador
    backtester = Backtester(telegram_notifier=notifier)
    
    # Enviar mensaje de inicio
    await notifier.send_message("🤖 Iniciando proceso de backtesting...")
    
    # Ejecutar backtests
    results = await backtester.run_multiple_backtests(symbols, timeframes, days, initial_balance)
    
    # Generar y enviar reporte
    await backtester.generate_report(results)
    
    # Generar y enviar gráficos
    await backtester.plot_results(results)
    
    # Enviar mensaje de finalización
    await notifier.send_message("✅ Proceso de backtesting completado")

# Ejecutar la función principal
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
