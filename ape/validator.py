import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ValidationResult:
    passed: bool
    law: str
    message: str
    error_magnitude: float = 0.0


class PhysicsValidator:
    def __init__(self, tolerance: float = 0.01):
        self.tolerance = tolerance
        self.violations = []
    
    def validate_energy_conservation(
        self, 
        mass: float, 
        velocity_before: np.ndarray, 
        velocity_after: np.ndarray,
        height_before: float,
        height_after: float,
        elasticity: float,
        gravity: float = 9.8
    ) -> ValidationResult:
        """
        Check if energy is conserved (or properly lost) in collision.
        For inelastic collision, energy should decrease by (1 - e^2).
        """
        ke_before = 0.5 * mass * np.dot(velocity_before, velocity_before)
        pe_before = mass * gravity * height_before
        total_before = ke_before + pe_before
        
        ke_after = 0.5 * mass * np.dot(velocity_after, velocity_after)
        pe_after = mass * gravity * height_after
        total_after = ke_after + pe_after
        
        expected_energy = total_before * (elasticity ** 2)
        
        error = abs(total_after - expected_energy) / total_before if total_before > 0 else 0
        
        passed = error <= self.tolerance
        
        result = ValidationResult(
            passed=passed,
            law="Energy Conservation",
            message=f"Energy before: {total_before:.3f}J, after: {total_after:.3f}J, expected: {expected_energy:.3f}J (error: {error*100:.2f}%)",
            error_magnitude=error
        )
        
        if not passed:
            self.violations.append(result)
        
        return result
    
    def validate_momentum_conservation(
        self,
        mass1: float,
        velocity1_before: np.ndarray,
        velocity1_after: np.ndarray,
        mass2: float = float('inf'),
        velocity2_before: np.ndarray = np.array([0.0, 0.0]),
        velocity2_after: np.ndarray = np.array([0.0, 0.0])
    ) -> ValidationResult:
        """
        Check momentum conservation. For collision with infinite mass (floor),
        momentum is not conserved (floor absorbs momentum).
        """
        if mass2 == float('inf'):
            return ValidationResult(
                passed=True,
                law="Momentum Conservation",
                message="Collision with infinite mass - momentum not conserved (expected)",
                error_magnitude=0.0
            )
        
        momentum_before = mass1 * velocity1_before + mass2 * velocity2_before
        momentum_after = mass1 * velocity1_after + mass2 * velocity2_after
        
        error = np.linalg.norm(momentum_after - momentum_before)
        total_momentum = np.linalg.norm(momentum_before)
        
        relative_error = error / total_momentum if total_momentum > 0 else 0
        
        passed = relative_error <= self.tolerance
        
        result = ValidationResult(
            passed=passed,
            law="Momentum Conservation",
            message=f"Momentum before: {momentum_before}, after: {momentum_after} (error: {relative_error*100:.2f}%)",
            error_magnitude=relative_error
        )
        
        if not passed:
            self.violations.append(result)
        
        return result
    
    def validate_velocity_direction(
        self,
        velocity_before: np.ndarray,
        velocity_after: np.ndarray,
        surface_normal: np.ndarray
    ) -> ValidationResult:
        """
        Check if velocity reverses correctly relative to surface normal.
        Perpendicular component should reverse, parallel should remain.
        """
        normal = surface_normal / np.linalg.norm(surface_normal)
        
        v_perp_before = np.dot(velocity_before, normal)
        v_perp_after = np.dot(velocity_after, normal)
        
        if v_perp_before < 0:
            expected_sign = 1
            actual_sign = np.sign(v_perp_after)
            
            passed = actual_sign == expected_sign
            
            result = ValidationResult(
                passed=passed,
                law="Velocity Direction",
                message=f"Perpendicular velocity before: {v_perp_before:.3f}, after: {v_perp_after:.3f} (should reverse)",
                error_magnitude=0.0 if passed else 1.0
            )
            
            if not passed:
                self.violations.append(result)
            
            return result
        
        return ValidationResult(
            passed=True,
            law="Velocity Direction",
            message="No collision detected (velocity not toward surface)",
            error_magnitude=0.0
        )
    
    def validate_elasticity_coefficient(
        self,
        velocity_before: np.ndarray,
        velocity_after: np.ndarray,
        surface_normal: np.ndarray,
        elasticity: float
    ) -> ValidationResult:
        """
        Check if velocity magnitude scales correctly by elasticity coefficient.
        """
        normal = surface_normal / np.linalg.norm(surface_normal)
        
        v_perp_before = abs(np.dot(velocity_before, normal))
        v_perp_after = abs(np.dot(velocity_after, normal))
        
        if v_perp_before > 0:
            expected_v_perp = v_perp_before * elasticity
            error = abs(v_perp_after - expected_v_perp) / v_perp_before
            
            passed = error <= self.tolerance
            
            result = ValidationResult(
                passed=passed,
                law="Elasticity Coefficient",
                message=f"Perpendicular velocity: {v_perp_before:.3f} -> {v_perp_after:.3f}, expected: {expected_v_perp:.3f} (error: {error*100:.2f}%)",
                error_magnitude=error
            )
            
            if not passed:
                self.violations.append(result)
            
            return result
        
        return ValidationResult(
            passed=True,
            law="Elasticity Coefficient",
            message="No collision detected",
            error_magnitude=0.0
        )
    
    def validate_collision(
        self,
        mass: float,
        velocity_before: np.ndarray,
        velocity_after: np.ndarray,
        position_before: np.ndarray,
        position_after: np.ndarray,
        elasticity: float,
        surface_normal: np.ndarray = np.array([0.0, 1.0]),
        gravity: float = 9.8
    ) -> List[ValidationResult]:
        """
        Run all validation checks for a collision event.
        """
        results = []
        
        results.append(self.validate_velocity_direction(
            velocity_before, velocity_after, surface_normal
        ))
        
        results.append(self.validate_elasticity_coefficient(
            velocity_before, velocity_after, surface_normal, elasticity
        ))
        
        results.append(self.validate_energy_conservation(
            mass, velocity_before, velocity_after,
            position_before[1], position_after[1],
            elasticity, gravity
        ))
        
        return results
    
    def get_summary(self) -> Dict:
        """
        Get summary of all validation results.
        """
        return {
            'total_violations': len(self.violations),
            'violations_by_law': {
                law: len([v for v in self.violations if v.law == law])
                for law in set(v.law for v in self.violations)
            },
            'violations': self.violations
        }
    
    def print_summary(self):
        """
        Print human-readable summary of validation results.
        """
        if not self.violations:
            print("\n✓ All physics validations passed!")
            return
        
        print(f"\n⚠ {len(self.violations)} physics violations detected:")
        
        for law, count in self.get_summary()['violations_by_law'].items():
            print(f"\n{law}: {count} violations")
            law_violations = [v for v in self.violations if v.law == law]
            for v in law_violations[:3]:
                print(f"  - {v.message}")
            if len(law_violations) > 3:
                print(f"  ... and {len(law_violations) - 3} more")
