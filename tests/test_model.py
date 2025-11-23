"""
Unit tests for Kiyotaki-Wright model
"""
import pytest
import numpy as np
import sys
sys.path.append('..')

from kw_model import KWEconomy, KWAnalyzer, Agent
from kw_model.strategies import FundamentalStrategy, SpeculativeStrategy


class TestAgent:
    """Test Agent class."""
    
    def test_agent_creation(self):
        """Test that agents are created correctly."""
        agent = Agent(
            agent_id=0,
            agent_type=1,
            production_good=2,
            consumption_good=1,
            storage_costs={1: 0.5, 2: 1.0, 3: 2.0},
            utility=10.0,
            disutility=2.0,
            initial_inventory=2
        )
        
        assert agent.agent_id == 0
        assert agent.agent_type == 1
        assert agent.production_good == 2
        assert agent.consumption_good == 1
        assert agent.inventory == 2
    
    def test_fundamental_strategy(self):
        """Test fundamental trading strategy."""
        strategy = FundamentalStrategy()
        
        agent = Agent(
            agent_id=0,
            agent_type=1,
            production_good=2,
            consumption_good=1,
            storage_costs={1: 0.5, 2: 1.0, 3: 2.0},
            utility=10.0,
            disutility=2.0,
            initial_inventory=2,
            strategy=strategy
        )
        
        # Should always want consumption good
        assert agent.wants_to_trade(2, 1) == True
        
        # Should not trade away consumption good
        assert agent.wants_to_trade(1, 2) == False
        
        # Should prefer lower storage cost
        assert agent.wants_to_trade(3, 2) == True  # 3 has higher cost than 2
        assert agent.wants_to_trade(2, 3) == False
    
    def test_consumption_production(self):
        """Test consumption and production."""
        agent = Agent(
            agent_id=0,
            agent_type=1,
            production_good=2,
            consumption_good=1,
            storage_costs={1: 0.5, 2: 1.0, 3: 2.0},
            utility=10.0,
            disutility=2.0,
            initial_inventory=1  # Has consumption good
        )
        
        # Consume and produce
        u, d = agent.consume_and_produce()
        
        assert u == 10.0  # Got utility
        assert d == 2.0   # Incurred disutility
        assert agent.inventory == 2  # Now has production good
        assert agent.consumption_count == 1
        assert agent.production_count == 1


class TestEconomy:
    """Test KWEconomy class."""
    
    def test_economy_creation_model_a(self):
        """Test economy creation for Model A."""
        economy = KWEconomy(
            model_type='A',
            num_agents=30,
            random_seed=42
        )
        
        assert len(economy.agents) == 30
        assert economy.model_type == 'A'
        
        # Check production mapping
        type1_agents = [a for a in economy.agents if a.agent_type == 1]
        assert all(a.production_good == 2 for a in type1_agents)
        assert all(a.consumption_good == 1 for a in type1_agents)
    
    def test_economy_creation_model_b(self):
        """Test economy creation for Model B."""
        economy = KWEconomy(
            model_type='B',
            num_agents=30,
            random_seed=42
        )
        
        assert len(economy.agents) == 30
        assert economy.model_type == 'B'
        
        # Check production mapping
        type1_agents = [a for a in economy.agents if a.agent_type == 1]
        assert all(a.production_good == 3 for a in type1_agents)
    
    def test_run_period(self):
        """Test running a single period."""
        economy = KWEconomy(
            model_type='A',
            num_agents=30,
            random_seed=42
        )
        
        initial_period = economy.current_period
        stats = economy.run_period()
        
        assert economy.current_period == initial_period + 1
        assert 'num_trades' in stats
        assert 'consumption_utility' in stats
        assert 'storage_costs' in stats
    
    def test_inventory_distribution(self):
        """Test inventory distribution calculation."""
        economy = KWEconomy(
            model_type='A',
            num_agents=30,
            random_seed=42
        )
        
        dist = economy.get_inventory_distribution()
        
        # Should be a dict with (agent_type, good) keys
        assert isinstance(dist, dict)
        
        # Proportions should sum to 1 for each type
        for agent_type in [1, 2, 3]:
            total = sum(
                prop for (t, g), prop in dist.items() if t == agent_type
            )
            assert abs(total - 1.0) < 0.01
    
    def test_run_simulation(self):
        """Test running full simulation."""
        economy = KWEconomy(
            model_type='A',
            num_agents=30,
            random_seed=42
        )
        
        results = economy.run_simulation(num_periods=100, burn_in=20)
        
        assert 'avg_trades_per_period' in results
        assert 'steady_state_distribution' in results
        assert 'welfare_by_type' in results
        assert economy.current_period == 100
    
    def test_fiat_money(self):
        """Test fiat money initialization."""
        economy = KWEconomy(
            model_type='A',
            num_agents=30,
            fiat_money=True,
            fiat_proportion=0.3,
            random_seed=42
        )
        
        # Check that some agents hold fiat money (good 0)
        fiat_holders = [a for a in economy.agents if a.inventory == 0]
        expected_fiat = int(30 * 0.3)
        
        assert len(fiat_holders) == expected_fiat


class TestAnalyzer:
    """Test KWAnalyzer class."""
    
    def test_compute_stock(self):
        """Test stock computation."""
        economy = KWEconomy(
            model_type='A',
            num_agents=30,
            random_seed=42
        )
        
        analyzer = KWAnalyzer(economy)
        
        # Sum of all stocks should be ~1.0
        stocks = analyzer.compute_all_stocks()
        total_stock = sum(stocks.values())
        
        assert abs(total_stock - 1.0) < 0.01
    
    def test_velocity_computation(self):
        """Test velocity computation."""
        economy = KWEconomy(
            model_type='A',
            num_agents=30,
            random_seed=42
        )
        
        # Run some periods to generate trades
        economy.run_simulation(200, burn_in=50)
        
        analyzer = KWAnalyzer(economy)
        velocities = analyzer.compute_all_velocities()
        
        # Velocities should be non-negative
        for good, vel in velocities.items():
            assert vel >= 0
    
    def test_equilibrium_identification(self):
        """Test equilibrium type identification."""
        economy = KWEconomy(
            model_type='A',
            num_agents=30,
            strategy_name='fundamental',
            random_seed=42
        )
        
        economy.run_simulation(500, burn_in=100)
        
        analyzer = KWAnalyzer(economy)
        eq_type = analyzer.compute_equilibrium_type()
        
        assert isinstance(eq_type, str)
        assert 'Equilibrium' in eq_type


def test_model_consistency():
    """Test that model results are consistent across runs with same seed."""
    economy1 = KWEconomy(model_type='A', num_agents=30, random_seed=42)
    economy1.run_simulation(100, burn_in=20)
    dist1 = economy1.get_inventory_distribution()
    
    economy2 = KWEconomy(model_type='A', num_agents=30, random_seed=42)
    economy2.run_simulation(100, burn_in=20)
    dist2 = economy2.get_inventory_distribution()
    
    # Should get same results with same seed
    for key in dist1.keys():
        assert abs(dist1[key] - dist2.get(key, 0)) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
