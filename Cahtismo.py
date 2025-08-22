import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configurar logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

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
    
    def detect_double_top(self, df, tolerance=0.02):
        if len(df) < 20:
            return False
        
        # Buscar dos picos cercanos en precio
        highs = df['High'].values
        peaks = []
        
        for i in range(1, len(highs)-1):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                peaks.append((i, highs[i]))
        
        if len(peaks) < 2:
            return False
        
        # Comparar los dos últimos picos
        last_two_peaks = sorted(peaks[-2:], key=lambda x: x[0])
        price_diff = abs(last_two_peaks[0][1] - last_two_peaks[1][1]) / last_two_peaks[0][1]
        
        if price_diff < tolerance:
            # Verificar que hay un valle entre ellos y ruptura del neckline
            valley = min(df['Low'][last_two_peaks[0][0]:last_two_peaks[1][0]])
            current_price = df['Close'].iloc[-1]
            
            if current_price < valley:
                return True
        
        return False
    
    def detect_double_bottom(self, df, tolerance=0.02):
        if len(df) < 20:
            return False
        
        # Buscar dos valles cercanos en precio
        lows = df['Low'].values
        valleys = []
        
        for i in range(1, len(lows)-1):
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                valleys.append((i, lows[i]))
        
        if len(valleys) < 2:
            return False
        
        # Comparar los dos últimos valles
        last_two_valleys = sorted(valleys[-2:], key=lambda x: x[0])
        price_diff = abs(last_two_valleys[0][1] - last_two_valleys[1][1]) / last_two_valleys[0][1]
        
        if price_diff < tolerance:
            # Verificar que hay un pico entre ellos y ruptura del neckline
            peak = max(df['High'][last_two_valleys[0][0]:last_two_valleys[1][0]])
            current_price = df['Close'].iloc[-1]
            
            if current_price > peak:
                return True
        
        return False
    
    def detect_head_shoulders(self, df):
        if len(df) < 30:
            return False
        
        highs = df['High'].values
        
        # Buscar patrón de tres picos: hombro izquierdo, cabeza, hombro derecho
        peaks = []
        for i in range(1, len(highs)-1):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                peaks.append((i, highs[i]))
        
        if len(peaks) < 3:
            return False
        
        # Verificar que el pico central sea el más alto
        if peaks[-2][1] > peaks[-3][1] and peaks[-2][1] > peaks[-1][1]:
            # Verificar que los hombros sean de altura similar
            shoulders_diff = abs(peaks[-3][1] - peaks[-1][1]) / peaks[-3][1]
            if shoulders_diff < 0.05:  # 5% de diferencia
                # Verificar ruptura del neckline
                neckline = min(df['Low'][peaks[-3][0]:peaks[-1][0]])
                current_price = df['Close'].iloc[-1]
                
                if current_price < neckline:
                    return True
        
        return False
    
    def detect_inverse_head_shoulders(self, df):
        if len(df) < 30:
            return False
        
        lows = df['Low'].values
        
        # Buscar patrón de tres valles: hombro izquierdo, cabeza, hombro derecho
        valleys = []
        for i in range(1, len(lows)-1):
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                valleys.append((i, lows[i]))
        
        if len(valleys) < 3:
            return False
        
        # Verificar que el valle central sea el más baja
        if valleys[-2][1] < valleys[-3][1] and valleys[-2][1] < valleys[-1][1]:
            # Verificar que los hombros sean de altura similar
            shoulders_diff = abs(valleys[-3][1] - valleys[-1][1]) / valleys[-3][1]
            if shoulders_diff < 0.05:  # 5% de diferencia
                # Verificar ruptura del neckline
                neckline = max(df['High'][valleys[-3][0]:valleys[-1][0]])
                current_price = df['Close'].iloc[-1]
                
                if current_price > neckline:
                    return True
        
        return False
    
    def detect_triangle(self, df):
        if len(df) < 20:
            return False
        
        # Detectar triángulo simétrico (máximos descendentes y mínimos ascendentes)
        highs = df['High'].tail(15).values
        lows = df['Low'].tail(15).values
        
        # Calcular tendencias
        high_trend = np.polyfit(range(len(highs)), highs, 1)[0]
        low_trend = np.polyfit(range(len(lows)), lows, 1)[0]
        
        # Triángulo simétrico: máximos descendentes y mínimos ascendentes
        if high_trend < 0 and low_trend > 0:
            return True
            
        return False
    
    def detect_rising_wedge(self, df):
        if len(df) < 20:
            return False
        
        # Detectar cuña ascendente (ambas líneas con pendiente positiva pero convergiendo)
        highs = df['High'].tail(15).values
        lows = df['Low'].tail(15).values
        
        # Calcular tendencias
        high_trend = np.polyfit(range(len(highs)), highs, 1)[0]
        low_trend = np.polyfit(range(len(lows)), lows, 1)[0]
        
        # Cuña ascendente: ambas líneas con pendiente positiva
        if high_trend > 0 and low_trend > 0:
            # Verificar convergencia (la pendiente de máximos es menor que la de mínimos)
            if high_trend < low_trend:
                return True
            
        return False
    
    def detect_falling_wedge(self, df):
        if len(df) < 20:
            return False
        
        # Detectar cuña descendente (ambas líneas con pendiente negativa pero convergiendo)
        highs = df['High'].tail(15).values
        lows = df['Low'].tail(15).values
        
        # Calcular tendencias
        high_trend = np.polyfit(range(len(highs)), highs, 1)[0]
        low_trend = np.polyfit(range(len(lows)), lows, 1)[0]
        
        # Cuña descendente: ambas líneas con pendiente negativa
        if high_trend < 0 and low_trend < 0:
            # Verificar convergencia (la pendiente de mínimos es menor que la de máximos)
            if low_trend < high_trend:
                return True
            
        return False
    
    def detect_bullish_flag(self, df):
        if len(df) < 20:
            return False
        
        # Detectar bandera alcista (pequeño consolidación después de un fuerte movimiento alcista)
        prices = df['Close'].values
        
        # Verificar movimiento alcista previo
        prev_move = prices[-15] - prices[-20]
        if prev_move <= 0:
            return False
            
        # Verificar consolidación (bandera)
        flag_high = max(prices[-10:])
        flag_low = min(prices[-10:])
        flag_range = flag_high - flag_low
        
        # La bandera debería ser una consolidación con rango limitado
        if flag_range / prices[-10] < 0.05:  # Menos del 5% de rango
            return True
            
        return False
    
    def detect_bearish_flag(self, df):
        if len(df) < 20:
            return False
        
        # Detectar bandera bajista (pequeño consolidación después de un fuerte movimiento bajista)
        prices = df['Close'].values
        
        # Verificar movimiento bajista previo
        prev_move = prices[-15] - prices[-20]
        if prev_move >= 0:
            return False
            
        # Verificar consolidación (bandera)
        flag_high = max(prices[-10:])
        flag_low = min(prices[-10:])
        flag_range = flag_high - flag_low
        
        # La bandera debería ser una consolidación con rango limitado
        if flag_range / prices[-10] < 0.05:  # Menos del 5% de rango
            return True
            
        return False

class Backtester:
    def __init__(self):
        self.analyzer = PatternAnalyzer()
        self.results = []
    
    def download_historical_data(self, symbol, start_date, end_date, interval='1h'):
        """Descarga datos históricos de Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            return df
        except Exception as e:
            print(f"Error descargando datos para {symbol}: {e}")
            return None
    
    def run_backtest(self, symbol, start_date, end_date, timeframe, initial_balance=10000):
        """Ejecuta backtesting para un símbolo y período específico"""
        print(f"Ejecutando backtest para {symbol} ({timeframe}) desde {start_date} hasta {end_date}")
        
        # Descargar datos históricos
        df = self.download_historical_data(symbol, start_date, end_date, timeframe)
        if df is None or df.empty:
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
                    print(f"{current_date}: COMPRA a {current_price} por patrones {bullish_patterns}")
                
                elif bearish_patterns and False:  # Desactivado para solo operar largas por ahora
                    # Entrar en posición corta (si decides operar en corto)
                    position = -1
                    entry_price = current_price
                    trades.append({
                        'date': current_date,
                        'action': 'SELL',
                        'price': current_price,
                        'balance': balance,
                        'patterns': bearish_patterns
                    })
                    print(f"{current_date}: VENTA a {current_price} por patrones {bearish_patterns}")
            
            # Salir de la posición (estrategia simple: salir después de 5 velas o con 5% de ganancia/pérdida)
            elif position != 0:
                holding_period = 5  # Salir después de 5 velas
                profit_target = 0.05  # 5% de objetivo de ganancia
                stop_loss = 0.03  # 3% de stop loss
                
                # Calcular ganancia/pérdida
                if position == 1:  # Posición larga
                    profit_pct = (current_price - entry_price) / entry_price
                    exit_condition = (profit_pct >= profit_target or 
                                     profit_pct <= -stop_loss or 
                                     len(trades) > 0 and (current_date - trades[-1]['date']).total_seconds() / (60*60) >= holding_period)
                
                elif position == -1:  # Posición corta
                    profit_pct = (entry_price - current_price) / entry_price
                    exit_condition = (profit_pct >= profit_target or 
                                     profit_pct <= -stop_loss or 
                                     len(trades) > 0 and (current_date - trades[-1]['date']).total_seconds() / (60*60) >= holding_period)
                
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
                    print(f"{current_date}: {action} a {current_price} - Balance: {balance:.2f} ({profit_pct*100:.2f}%)")
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
            'start_date': start_date,
            'end_date': end_date,
            'initial_balance': initial_balance,
            'final_balance': final_balance,
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'trades': trades
        }
        
        self.results.append(result)
        return result
    
    def run_multiple_backtests(self, symbols, timeframes, start_date, end_date, initial_balance=10000):
        """Ejecuta backtests para múltiples símbolos y timeframes"""
        all_results = []
        
        for symbol in symbols:
            for timeframe in timeframes:
                result = self.run_backtest(symbol, start_date, end_date, timeframe, initial_balance)
                if result:
                    all_results.append(result)
        
        return all_results
    
    def generate_report(self, results):
        """Genera un reporte de los resultados del backtesting"""
        if not results:
            print("No hay resultados para reportar")
            return
        
        print("\n" + "="*80)
        print("REPORTE DE BACKTESTING - PATRONES CHARTISTAS")
        print("="*80)
        
        for result in results:
            print(f"\nSímbolo: {result['symbol']} | Timeframe: {result['timeframe']}")
            print(f"Período: {result['start_date']} to {result['end_date']}")
            print(f"Balance inicial: ${result['initial_balance']:.2f}")
            print(f"Balance final: ${result['final_balance']:.2f}")
            print(f"Retorno total: {result['total_return']:.2f}%")
            print(f"Número de operaciones: {result['num_trades']}")
            print(f"Ratio de operaciones ganadoras: {result['win_rate']:.2f}%")
            print("-" * 40)
        
        # Calcular promedios
        avg_return = np.mean([r['total_return'] for r in results])
        avg_win_rate = np.mean([r['win_rate'] for r in results])
        
        print(f"\nRESUMEN GENERAL:")
        print(f"Retorno promedio: {avg_return:.2f}%")
        print(f"Ratio de acierto promedio: {avg_win_rate:.2f}%")
        print("="*80)
    
    def plot_results(self, results):
        """Genera gráficos de los resultados"""
        if not results:
            return
        
        # Gráfico de balances
        plt.figure(figsize=(12, 6))
        for result in results:
            # Extraer balances a lo largo del tiempo
            balances = [result['initial_balance']]
            dates = [result['start_date']]
            
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
        plt.show()
        
        # Gráfico de distribución de retornos
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
        plt.show()

# Ejemplo de uso
if __name__ == "__main__":
    # Configuración
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
    timeframes = ["15m", "1h"]
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    initial_balance = 10000
    
    # Crear backtester y ejecutar pruebas
    backtester = Backtester()
    results = backtester.run_multiple_backtests(symbols, timeframes, start_date, end_date, initial_balance)
    
    # Generar reporte y gráficos
    backtester.generate_report(results)
    backtester.plot_results(results)
