import sys
sys.path.append("../")

from typing import Any
import numpy as np
from openfermion.transforms import jordan_wigner
from quri_parts.algo.ansatz import SymmetryPreservingReal
from quri_parts.algo.optimizer import SPSA, OptimizerStatus
from quri_parts.circuit import LinearMappedUnboundParametricQuantumCircuit
from quri_parts.core.estimator.gradient import parameter_shift_gradient_estimates
from quri_parts.core.measurement import bitwise_commuting_pauli_measurement, CachedMeasurementFactory
from quri_parts.core.sampling.shots_allocator import create_equipartition_shots_allocator
from quri_parts.core.state import ParametricCircuitQuantumState, ComputationalBasisState
from quri_parts.openfermion.operator import operator_from_openfermion_op
from utils.challenge_2024 import ChallengeSampling, ExceededError, problem_hamiltonian

from quri_parts.core.operator import *
from scipy.optimize import minimize
def n_qubit_truncate(hamiltonian, q):
    ham = Operator()
    isempty = False
    for key in hamiltonian:
        if (q == 1 and (len(key) == 1 or len(key) == 0)) or len(key) == q:
            ham[pauli_label(key)] = hamiltonian[pauli_label(key)]
    if len(ham) == 0: 
        isempty = True
    return ham, isempty

challenge_sampling = ChallengeSampling()

def cost_fn(hamiltonian, parametric_state, param_values, estimator):
    estimate = estimator(hamiltonian, parametric_state, [param_values])
    return estimate[0].value.real



def vqe(hamiltonian, parametric_state, estimator, init_params, optimizer):
    opt_state = optimizer.get_init_state(init_params)
    energy_history = []

    def c_fn(param_values):
        return cost_fn(hamiltonian, parametric_state, param_values, estimator)

    while True:
        try:
            opt_state = optimizer.step(opt_state, c_fn)
            energy_history.append(opt_state.cost)
        except ExceededError as e:
            print(str(e))
            print(opt_state.cost)
            return opt_state, energy_history

        if opt_state.status == OptimizerStatus.FAILED:
            print("Optimizer failed")
            break
        if opt_state.status == OptimizerStatus.CONVERGED:
            print("Optimizer converged")
            break
    return opt_state, energy_history
    
def apply_gate(circuit, gate, qubit):
    if gate == 'rx':
        circuit.add_RX_gate(qubit, np.random.rand())
    elif gate == 'ry':
        circuit.add_RY_gate(qubit, np.random.rand())
    elif gate == 'rz':
        circuit.add_RZ_gate(qubit, np.random.rand())
    return circuit

def apply_cnot(circuit, control_qubit, target_qubit):
    circuit.add_CNOT_gate(control_qubit, target_qubit)
    return circuit

def apply_swap(circuit, qubit1, qubit2):
    circuit.add_SWAP_gate(qubit1, qubit2)
    return circuit
    
def iqcc(combined_hamiltonian,n_qubits,current_ansatz, current_params, estimator, max_iterations=100, convergence_threshold=1e-6):

    operator_pool = ['rx', 'ry', 'rz']
    new_ansatz = LinearMappedUnboundParametricQuantumCircuit(n_qubits)
    new_ansatz.extend(current_ansatz)
    new_params = np.copy(current_params)

    energy_history = []

    for iteration in range(max_iterations):
        best_energy = float('inf')
        
        best_gate = None
        best_qubits = None
        best_params = None
        optimizer = SPSA(ftol=10e-5)
        # Evaluate potential improvements by adding each gate from the operator pool
        for gate in operator_pool:
            if gate in ['rx', 'ry', 'rz']:
                for qubit in range(n_qubits):
                    parametric_state = ParametricCircuitQuantumState(n_qubits, new_ansatz)   
                    temp_ansatz = new_ansatz.get_mutable_copy()
                    temp_ansatz = apply_gate(temp_ansatz, gate, qubit)
                    temp_params = np.copy(new_params)
                    temp_res, energy_history = vqe(
                    combined_hamiltonian,
                    parametric_state,
                    estimator,
                    temp_params,
                    optimizer
                )
                    
                    if energy_history[-1] < best_energy:
                        best_energy = energy_history[-1]
                        best_gate = (gate, qubit)
                        best_params = temp_res.params
            elif gate == 'cnot':
                for control_qubit in range(n_qubits):
                    for target_qubit in range(n_qubits):
                        if control_qubit != target_qubit:
                            temp_ansatz = new_ansatz.get_mutable_copy()
                            temp_ansatz = apply_cnot(temp_ansatz, control_qubit, target_qubit)
                            temp_cost = cost_fn(new_params)
                            if temp_cost < best_cost:
                                best_cost = temp_cost
                                best_gate = (gate, control_qubit, target_qubit)

            
            elif gate == 'swap':
                for qubit1 in range(n_qubits):
                    for qubit2 in range(qubit1 + 1, n_qubits):
                        temp_ansatz = new_ansatz.get_mutable_copy()
                        temp_ansatz = apply_swap(temp_ansatz, qubit1, qubit2)
                        temp_cost = cost_fn(new_params)
                        if temp_cost < best_cost:
                            best_cost = best_energy
                            best_gate = (gate, qubit1, qubit2)
                            best_params = new_params

        # Apply the best gate to the new ansatz and update parameters
        if best_gate:
            if best_gate[0] in ['rx', 'ry', 'rz']:
                new_ansatz = apply_gate(new_ansatz, best_gate[0], best_gate[1])
            elif best_gate[0] == 'cnot':
                new_ansatz = apply_cnot(new_ansatz, best_gate[1], best_gate[2])
            elif best_gate[0] == 'swap':
                new_ansatz = apply_swap(new_ansatz, best_gate[1], best_gate[2])
            new_params = best_params
            energy_history.append(best_energy)

            if iteration > 0 and abs(energy_history[-1] - energy_history[-2]) < convergence_threshold:
                break

    return new_ansatz, new_params

class RunAlgorithm:
    def __init__(self) -> None:
        challenge_sampling.reset()

    def result_for_evaluation(self, seed: int, hamiltonian_directory: str) -> tuple[Any, float]:
        energy_final = self.get_result(seed, hamiltonian_directory)
        total_shots = challenge_sampling.total_shots
        return energy_final, total_shots

    def get_result(self, seed: int, hamiltonian_directory: str) -> float:

        energy_change_negligible_times = 0
        current_hamiltonian = Operator()
        qubit_ham_list = [n for n in range(2,29)]
        thresholds = np.arange(6.0, 0.0, -0.01)
        n_qubits = 12
        init_qubits = 2
        ham = problem_hamiltonian(n_qubits, seed, hamiltonian_directory)
        n_site = init_qubits // 2
        total_shots = 10
        jw_hamiltonian = jordan_wigner(ham)

        hamiltonian = operator_from_openfermion_op(jw_hamiltonian)
        shots_allocator = create_equipartition_shots_allocator()
        cached_measurement_factory = CachedMeasurementFactory(
            bitwise_commuting_pauli_measurement
        )

         # make hf + SPReal ansatz
        hf_gates = ComputationalBasisState(init_qubits, bits=2**n_site - 1).circuit.gates
        hf_circuit = LinearMappedUnboundParametricQuantumCircuit(init_qubits).combine(
            hf_gates
        )
        current_ansatz = SymmetryPreservingReal(qubit_count=init_qubits, reps=init_qubits)
        hf_circuit.extend(current_ansatz)
        shots_allocator = create_equipartition_shots_allocator()
        cached_measurement_factory = CachedMeasurementFactory(
            bitwise_commuting_pauli_measurement
        )

        

        sampling_estimator = challenge_sampling.create_concurrent_parametric_sampling_estimator(total_shots, cached_measurement_factory, shots_allocator)

        optimizer = SPSA(ftol=10e-5)

        init_params = np.random.rand(current_ansatz.parameter_count) * 2 * np.pi * 0.001
        current_params = init_params
        
        previous_energy = None

        
        for qubit in qubit_ham_list:
            
            req_hamiltonion, isempty = n_qubit_truncate(hamiltonian, qubit)
            if isempty == False:
                if qubit >= 3:
                    new_circuit = LinearMappedUnboundParametricQuantumCircuit(qubit)
                    print(new_circuit)
                    new_circuit = new_ansatz.get_mutable_copy()
                    new_param_index = new_circuit.add_RY_gate(qubit, np.random.rand())
                    current_ansatz = new_circuit

                elif qubit != 2:
                    break
            for threshold in thresholds:
                # Add truncated higher-order interactions
                truncated_hamiltonian = truncate(req_hamiltonion, threshold)
                    
                # Combine with current Hamiltonian
                combined_hamiltonian = current_hamiltonian + truncated_hamiltonian
                    
                # Solve using iQCC
                new_ansatz, new_params = iqcc(combined_hamiltonian,qubit,current_ansatz, current_params, sampling_estimator)
                parametric_state = ParametricCircuitQuantumState(qubit, new_ansatz)  
                
                # Optimize using VQE
                result, energy_history = vqe(combined_hamiltonian,parametric_state,sampling_estimator,new_params,optimizer)
    
                # Check for convergence (energy change)
                if previous_energy is not None:
                    energy_change = np.abs(previous_energy - energy_history[-1])
                    if energy_change < 1e-3:
                        energy_change_negligible_times += 1
                    if energy_change_negligible_times > 3:
                        break
                previous_energy = energy_history[-1]
                    
                # Update current Hamiltonian, ansatz, and parameters
                current_hamiltonian = combined_hamiltonian
                current_ansatz = new_ansatz
                current_params = new_params

          
        return min(energy_history)

  

if __name__ == "__main__":
    run_algorithm = RunAlgorithm()
    print(run_algorithm.get_result(seed=0, hamiltonian_directory="../hamiltonian"))


