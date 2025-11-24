"""
Tax Scenario Calculator
Feature 6: Interactive tax calculations for Dutch tax system (IB, VPB, BTW, Erfbelasting)
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class TaxType(str, Enum):
    """Types of Dutch taxes"""
    INCOME_TAX = "ib"  # Inkomstenbelasting
    CORPORATE_TAX = "vpb"  # Vennootschapsbelasting
    VAT = "btw"  # Omzetbelasting (BTW)
    INHERITANCE_TAX = "erfbelasting"
    GIFT_TAX = "schenkbelasting"


@dataclass
class TaxScenario:
    """Tax calculation scenario"""
    scenario_name: str
    parameters: Dict
    results: Dict
    total_tax: float
    effective_rate: float


class DutchTaxCalculator:
    """Calculator for Dutch tax scenarios (2024 rates)"""
    
    # Income Tax Brackets 2024 (Inkomstenbelasting)
    IB_BRACKETS_2024 = [
        {"max": 75518, "rate": 0.3693},  # Box 1: 36.93%
        {"max": float('inf'), "rate": 0.4950}  # Box 1: 49.50%
    ]
    
    # Corporate Tax Rates 2024 (VPB)
    VPB_RATE_LOW = 0.1901  # 19.01% up to €200k
    VPB_THRESHOLD = 200000
    VPB_RATE_HIGH = 0.2567  # 25.67% above €200k
    
    # VAT Rates 2024 (BTW)
    BTW_STANDARD = 0.21  # 21% standard rate
    BTW_REDUCED = 0.09   # 9% reduced rate
    BTW_ZERO = 0.00      # 0% zero rate
    
    # Inheritance Tax 2024 (Erfbelasting) - Partner/Children
    INHERITANCE_BRACKETS_PARTNER = [
        {"max": 770043, "rate": 0.10},
        {"max": float('inf'), "rate": 0.20}
    ]
    
    INHERITANCE_BRACKETS_CHILDREN = [
        {"max": 141223, "rate": 0.10},
        {"max": float('inf'), "rate": 0.20}
    ]
    
    # Tax-free allowances 2024
    INHERITANCE_EXEMPT_PARTNER = 770043
    INHERITANCE_EXEMPT_CHILD = 25439
    GIFT_EXEMPT_CHILD = 6035
    
    def calculate_income_tax(self, income: float, box1: bool = True) -> Dict:
        """
        Calculate Dutch income tax (Box 1)
        
        Args:
            income: Taxable income in euros
            box1: If True, use Box 1 rates (work income)
        
        Returns:
            Dict with breakdown and total tax
        """
        if income <= 0:
            return {"total_tax": 0, "effective_rate": 0, "breakdown": []}
        
        total_tax = 0
        breakdown = []
        remaining = income
        prev_max = 0
        
        for bracket in self.IB_BRACKETS_2024:
            bracket_max = bracket["max"]
            rate = bracket["rate"]
            
            if remaining <= 0:
                break
            
            taxable_in_bracket = min(remaining, bracket_max - prev_max)
            tax_in_bracket = taxable_in_bracket * rate
            total_tax += tax_in_bracket
            
            breakdown.append({
                "bracket": f"€{int(prev_max):,} - €{int(bracket_max):,}",
                "rate": f"{rate*100:.2f}%",
                "taxable": taxable_in_bracket,
                "tax": tax_in_bracket
            })
            
            remaining -= taxable_in_bracket
            prev_max = bracket_max
        
        return {
            "gross_income": income,
            "total_tax": total_tax,
            "net_income": income - total_tax,
            "effective_rate": (total_tax / income) * 100 if income > 0 else 0,
            "breakdown": breakdown
        }
    
    def calculate_corporate_tax(self, profit: float) -> Dict:
        """
        Calculate Dutch corporate tax (VPB)
        
        Args:
            profit: Taxable profit in euros
        
        Returns:
            Dict with breakdown and total tax
        """
        if profit <= 0:
            return {"total_tax": 0, "effective_rate": 0, "breakdown": []}
        
        breakdown = []
        
        # Low bracket (up to €200k)
        low_bracket_amount = min(profit, self.VPB_THRESHOLD)
        low_bracket_tax = low_bracket_amount * self.VPB_RATE_LOW
        breakdown.append({
            "bracket": f"€0 - €{self.VPB_THRESHOLD:,}",
            "rate": f"{self.VPB_RATE_LOW*100:.2f}%",
            "taxable": low_bracket_amount,
            "tax": low_bracket_tax
        })
        
        # High bracket (above €200k)
        high_bracket_tax = 0
        if profit > self.VPB_THRESHOLD:
            high_bracket_amount = profit - self.VPB_THRESHOLD
            high_bracket_tax = high_bracket_amount * self.VPB_RATE_HIGH
            breakdown.append({
                "bracket": f"€{self.VPB_THRESHOLD:,}+",
                "rate": f"{self.VPB_RATE_HIGH*100:.2f}%",
                "taxable": high_bracket_amount,
                "tax": high_bracket_tax
            })
        
        total_tax = low_bracket_tax + high_bracket_tax
        
        return {
            "gross_profit": profit,
            "total_tax": total_tax,
            "net_profit": profit - total_tax,
            "effective_rate": (total_tax / profit) * 100 if profit > 0 else 0,
            "breakdown": breakdown
        }
    
    def calculate_vat(self, net_amount: float, vat_rate: str = "standard") -> Dict:
        """
        Calculate Dutch VAT (BTW)
        
        Args:
            net_amount: Amount excluding VAT
            vat_rate: "standard" (21%), "reduced" (9%), or "zero" (0%)
        
        Returns:
            Dict with VAT calculation
        """
        rate_map = {
            "standard": self.BTW_STANDARD,
            "reduced": self.BTW_REDUCED,
            "zero": self.BTW_ZERO
        }
        
        rate = rate_map.get(vat_rate, self.BTW_STANDARD)
        vat_amount = net_amount * rate
        gross_amount = net_amount + vat_amount
        
        return {
            "net_amount": net_amount,
            "vat_rate": f"{rate*100:.0f}%",
            "vat_amount": vat_amount,
            "gross_amount": gross_amount
        }
    
    def calculate_inheritance_tax(self, inheritance: float, relationship: str = "child") -> Dict:
        """
        Calculate Dutch inheritance tax (Erfbelasting)
        
        Args:
            inheritance: Inheritance amount in euros
            relationship: "partner" or "child"
        
        Returns:
            Dict with tax calculation
        """
        if relationship == "partner":
            exempt = self.INHERITANCE_EXEMPT_PARTNER
            brackets = self.INHERITANCE_BRACKETS_PARTNER
        else:
            exempt = self.INHERITANCE_EXEMPT_CHILD
            brackets = self.INHERITANCE_BRACKETS_CHILDREN
        
        taxable = max(0, inheritance - exempt)
        
        if taxable <= 0:
            return {
                "inheritance": inheritance,
                "exempt_amount": exempt,
                "taxable_amount": 0,
                "total_tax": 0,
                "net_inheritance": inheritance,
                "effective_rate": 0,
                "breakdown": []
            }
        
        total_tax = 0
        breakdown = []
        remaining = taxable
        prev_max = 0
        
        for bracket in brackets:
            bracket_max = bracket["max"]
            rate = bracket["rate"]
            
            if remaining <= 0:
                break
            
            taxable_in_bracket = min(remaining, bracket_max - prev_max)
            tax_in_bracket = taxable_in_bracket * rate
            total_tax += tax_in_bracket
            
            breakdown.append({
                "bracket": f"€{int(prev_max):,} - €{int(bracket_max):,}",
                "rate": f"{rate*100:.0f}%",
                "taxable": taxable_in_bracket,
                "tax": tax_in_bracket
            })
            
            remaining -= taxable_in_bracket
            prev_max = bracket_max
        
        return {
            "inheritance": inheritance,
            "exempt_amount": exempt,
            "taxable_amount": taxable,
            "total_tax": total_tax,
            "net_inheritance": inheritance - total_tax,
            "effective_rate": (total_tax / inheritance) * 100 if inheritance > 0 else 0,
            "breakdown": breakdown
        }
    
    def compare_scenarios(self, scenarios: List[Dict]) -> Dict:
        """
        Compare multiple tax scenarios
        
        Args:
            scenarios: List of scenario parameters
        
        Returns:
            Comparison results with recommendations
        """
        results = []
        
        for idx, scenario in enumerate(scenarios):
            scenario_name = scenario.get("name", f"Scenario {idx+1}")
            tax_type = scenario.get("tax_type", "ib")
            
            if tax_type == "ib":
                calc_result = self.calculate_income_tax(scenario.get("income", 0))
            elif tax_type == "vpb":
                calc_result = self.calculate_corporate_tax(scenario.get("profit", 0))
            elif tax_type == "erfbelasting":
                calc_result = self.calculate_inheritance_tax(
                    scenario.get("inheritance", 0),
                    scenario.get("relationship", "child")
                )
            else:
                continue
            
            results.append({
                "scenario_name": scenario_name,
                "tax_type": tax_type,
                "total_tax": calc_result.get("total_tax", 0),
                "effective_rate": calc_result.get("effective_rate", 0),
                "details": calc_result
            })
        
        # Find best scenario (lowest tax)
        if results:
            best_scenario = min(results, key=lambda x: x["total_tax"])
            worst_scenario = max(results, key=lambda x: x["total_tax"])
            
            return {
                "scenarios": results,
                "best_scenario": best_scenario["scenario_name"],
                "best_tax": best_scenario["total_tax"],
                "worst_scenario": worst_scenario["scenario_name"],
                "worst_tax": worst_scenario["total_tax"],
                "savings": worst_scenario["total_tax"] - best_scenario["total_tax"]
            }
        
        return {"scenarios": [], "best_scenario": None}


# Global calculator instance
tax_calculator = DutchTaxCalculator()
