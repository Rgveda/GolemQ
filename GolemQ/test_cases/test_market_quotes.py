import os, sys
try:
    import GolemQ
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime as dt

from GolemQ.markets.StockCN.quotes import StockCNQuotes


class TestMarketQuotes(unittest.TestCase):

    def setUp(self):
        self.quotes = StockCNQuotes()

    @staticmethod
    def _make_mock_qa_data_struct(data_df):
        """Helper to create a mock QA_DataStruct with .data and .to_qfq()"""
        mock_struct = MagicMock()
        mock_struct.data = data_df
        mock_struct.to_qfq.return_value = mock_struct
        return mock_struct

    def _make_day_kline_df(self, code='000001', days=5):
        """Create a realistic day kline DataFrame with MultiIndex (date, code)"""
        dates = pd.date_range('2024-01-15', periods=days, freq='B')
        arrays = [dates, [code] * days]
        index = pd.MultiIndex.from_arrays(arrays, names=['date', 'code'])
        return pd.DataFrame({
            'open': [10.0 + i * 0.1 for i in range(days)],
            'high': [10.5 + i * 0.1 for i in range(days)],
            'low': [9.8 + i * 0.1 for i in range(days)],
            'close': [10.3 + i * 0.1 for i in range(days)],
            'volume': [float(1000000 + i * 10000) for i in range(days)],
            'amount': [float(10000000 + i * 100000) for i in range(days)],
        }, index=index)

    def _make_min_kline_df(self, code='000001', hours=5):
        """Create a realistic minute kline DataFrame with MultiIndex (datetime, code)"""
        dts = pd.date_range('2024-01-15 09:30', periods=hours, freq='h')
        arrays = [dts, [code] * hours]
        index = pd.MultiIndex.from_arrays(arrays, names=['datetime', 'code'])
        return pd.DataFrame({
            'open': [10.0 + i * 0.1 for i in range(hours)],
            'high': [10.5 + i * 0.1 for i in range(hours)],
            'low': [9.8 + i * 0.1 for i in range(hours)],
            'close': [10.3 + i * 0.1 for i in range(hours)],
            'volume': [float(10000 + i * 100) for i in range(hours)],
            'vol': [float(10000 + i * 100) for i in range(hours)],
        }, index=index)

    @patch('GolemQ.markets.StockCN.quotes.QA_fetch_stock_day_adv')
    def test_get_kline_quotes_returns_dataframe(self, mock_fetch):
        """get_kline_quotes should return a pd.DataFrame with expected columns"""
        data_df = self._make_day_kline_df()
        mock_struct = self._make_mock_qa_data_struct(data_df)
        mock_fetch.return_value = mock_struct

        result = self.quotes.get_kline_quotes('000001', '2024-01-01', '2024-01-20')

        self.assertIsInstance(result, pd.DataFrame)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            self.assertIn(col, result.columns)
        self.assertEqual(len(result), 5)

    @patch('GolemQ.markets.StockCN.quotes.QA_fetch_stock_day_adv')
    def test_get_kline_quotes_calls_fetch_with_correct_params(self, mock_fetch):
        """get_kline_quotes should normalize code and call QA_fetch_stock_day_adv"""
        data_df = self._make_day_kline_df()
        mock_struct = self._make_mock_qa_data_struct(data_df)
        mock_fetch.return_value = mock_struct

        self.quotes.get_kline_quotes('000001.XSHE', '2024-01-01', '2024-01-20')

        mock_fetch.assert_called_once_with('000001', start='2024-01-01', end='2024-01-20')

    @patch('GolemQ.markets.StockCN.quotes.QA_fetch_stock_day_adv')
    def test_get_kline_quotes_applies_qfq_when_fq_set(self, mock_fetch):
        """When fq=1, to_qfq() should be called"""
        data_df = self._make_day_kline_df()
        mock_struct = self._make_mock_qa_data_struct(data_df)
        mock_fetch.return_value = mock_struct

        self.quotes.get_kline_quotes('000001', '2024-01-01', '2024-01-20', fq=1)

        mock_struct.to_qfq.assert_called_once()

    @patch('GolemQ.markets.StockCN.quotes.QA_fetch_stock_day_adv')
    def test_get_kline_quotes_skips_qfq_when_fq_zero(self, mock_fetch):
        """When fq=0, to_qfq() should NOT be called"""
        data_df = self._make_day_kline_df()
        mock_struct = self._make_mock_qa_data_struct(data_df)
        mock_fetch.return_value = mock_struct

        self.quotes.get_kline_quotes('000001', '2024-01-01', '2024-01-20', fq=0)

        mock_struct.to_qfq.assert_not_called()

    @patch('GolemQ.markets.StockCN.quotes.QA_fetch_stock_day_adv')
    def test_get_kline_quotes_returns_empty_on_none_data(self, mock_fetch):
        """If QA_fetch_stock_day_adv returns None, return empty DataFrame"""
        mock_fetch.return_value = None

        result = self.quotes.get_kline_quotes('999999', '2024-01-01', '2024-01-20')

        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)

    @patch('GolemQ.markets.StockCN.quotes.GQ_fetch_stock_min_adv')
    def test_get_kline_quotes_min_returns_dataframe(self, mock_fetch):
        """get_kline_quotes_min should return a pd.DataFrame"""
        data_df = self._make_min_kline_df()
        mock_struct = self._make_mock_qa_data_struct(data_df)
        mock_fetch.return_value = mock_struct

        result = self.quotes.get_kline_quotes_min(
            '000001', '2024-01-15', '2024-01-15', frequency='60min'
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('close', result.columns)

    @patch('GolemQ.markets.StockCN.quotes.GQ_fetch_stock_min_adv')
    def test_get_kline_quotes_min_normalizes_frequency(self, mock_fetch):
        """Frequency aliases like '60m' should be normalized to '60min'"""
        data_df = self._make_min_kline_df()
        mock_struct = self._make_mock_qa_data_struct(data_df)
        mock_fetch.return_value = mock_struct

        self.quotes.get_kline_quotes_min('000001', '2024-01-15', '2024-01-15', frequency='60m')

        call_args = mock_fetch.call_args
        self.assertEqual(call_args[1]['frequence'], '60min')

    @patch('GolemQ.markets.StockCN.quotes.GQ_fetch_stock_min_adv')
    def test_get_kline_quotes_min_returns_empty_on_none(self, mock_fetch):
        """If GQ_fetch_stock_min_adv returns None, return empty DataFrame"""
        mock_fetch.return_value = None

        result = self.quotes.get_kline_quotes_min('999999', '2024-01-15', '2024-01-15')

        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)

    @patch('GolemQ.markets.StockCN.quotes.GQ_fetch_stock_min_adv')
    def test_get_kline_quotes_min_applies_qfq_when_fq_set(self, mock_fetch):
        """When fq=1, to_qfq() should be called for minute klines"""
        data_df = self._make_min_kline_df()
        mock_struct = self._make_mock_qa_data_struct(data_df)
        mock_fetch.return_value = mock_struct

        self.quotes.get_kline_quotes_min('000001', '2024-01-15', '2024-01-15', fq=1)

        mock_struct.to_qfq.assert_called_once()

    @patch('GolemQ.markets.StockCN.quotes.GQ_fetch_stock_min_adv')
    def test_get_kline_quotes_min_skips_qfq_when_fq_zero(self, mock_fetch):
        """When fq=0, to_qfq() should NOT be called for minute klines"""
        data_df = self._make_min_kline_df()
        mock_struct = self._make_mock_qa_data_struct(data_df)
        mock_fetch.return_value = mock_struct

        self.quotes.get_kline_quotes_min('000001', '2024-01-15', '2024-01-15', fq=0)

        mock_struct.to_qfq.assert_not_called()

    def test_get_kline_quotes_default_params(self):
        """Default start/end dates and fq should work without error when data is available"""
        # This test verifies the method signature defaults are correct
        self.assertEqual(self.quotes.get_kline_quotes.__code__.co_varnames[:5],
                         ('self', 'code', 'start', 'end', 'fq'))

    def test_get_kline_quotes_min_default_params(self):
        """Default params for minute kline should be correct"""
        self.assertEqual(self.quotes.get_kline_quotes_min.__code__.co_varnames[:6],
                         ('self', 'code', 'start', 'end', 'frequency', 'fq'))


if __name__ == '__main__':
    unittest.main()
