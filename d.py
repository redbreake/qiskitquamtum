import numpy as np
import matplotlib.pyplot as plt
from qiskit_algorithms.utils import algorithm_globals
from qiskit_algorithms import SamplingVQE
from qiskit_algorithms.optimizers import SPSA
from qiskit.circuit.library import RealAmplitudes
# 1. CAMBIO: Importar StatevectorSampler en lugar de Sampler
from qiskit.primitives import StatevectorSampler
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

class VehicleRoutingQuantumSolver:
    """
    Clase para resolver el Problema de Enrutamiento de Vehículos (VRP) usando Qiskit.
    Este código está corregido para funcionar con Qiskit 1.0+ (incluyendo 2.x).
    """
    def __init__(self, n_clients, n_vehicles, instance=None):
        self.n = n_clients + 1
        self.K = n_vehicles
        self.depot_node = 0
        
        if instance is not None:
            self.instance = instance
            self.xc, self.yc = None, None
        else:
            self.xc, self.yc, self.instance = self.generate_random_instance()

    def generate_random_instance(self):
        np.random.seed(1543)
        coords = (np.random.rand(self.n, 2) - 0.5) * 10
        xc, yc = coords[:, 0], coords[:, 1]
        
        instance = np.zeros([self.n, self.n])
        for i in range(self.n):
            for j in range(i + 1, self.n):
                dist = (xc[i] - xc[j])**2 + (yc[i] - yc[j])**2
                instance[i, j] = dist
                instance[j, i] = dist
        return xc, yc, instance

    def build_vrp_qp(self) -> QuadraticProgram:
        qp = QuadraticProgram(name="VehicleRoutingProblem")
        
        for i in range(self.n):
            for j in range(self.n):
                for k in range(self.K):
                    qp.binary_var(f"x_{i}_{j}_{k}")

        objective_linear = {f"x_{i}_{j}_{k}": self.instance[i, j] 
                            for i in range(self.n) for j in range(self.n) if i != j for k in range(self.K)}
        qp.minimize(linear=objective_linear)
        
        for j in range(1, self.n):
            constraint_linear = {f"x_{i}_{j}_{k}": 1 for i in range(self.n) if i != j for k in range(self.K)}
            qp.linear_constraint(linear=constraint_linear, sense="==", rhs=1, name=f"visit_client_{j}")
            
        for k in range(self.K):
            constraint_linear = {f"x_{self.depot_node}_{j}_{k}": 1 for j in range(1, self.n)}
            qp.linear_constraint(linear=constraint_linear, sense="==", rhs=1, name=f"vehicle_leaves_depot_{k}")
            
        for j in range(1, self.n):
            for k in range(self.K):
                sum_in = {f"x_{i}_{j}_{k}": 1 for i in range(self.n) if i != j}
                sum_out = {f"x_{j}_{i}_{k}": -1 for i in range(self.n) if i != j}
                qp.linear_constraint(linear={**sum_in, **sum_out}, sense="==", rhs=0, name=f"flow_conservation_{j}_{k}")

        return qp

    def solve(self, qp: QuadraticProgram):
        print("Iniciando la resolución cuántica (esto puede tardar unos minutos)...")
        
        algorithm_globals.random_seed = 10598
        
        # 2. CAMBIO: Usar StatevectorSampler
        sampler = StatevectorSampler()
        
        # 3. CAMBIO: Especificar el número de qubits para el ansatz
        num_qubits = qp.get_num_vars()
        ansatz = RealAmplitudes(num_qubits=num_qubits)
        
        optimizer = SPSA(maxiter=50)
        
        sampling_vqe = SamplingVQE(sampler=sampler, ansatz=ansatz, optimizer=optimizer)
        quantum_solver = MinimumEigenOptimizer(min_eigen_solver=sampling_vqe)
        
        result = quantum_solver.solve(qp)
        print("Resolución completada.")
        
        return result

    def visualize_solution(self, result):
        if self.xc is None or self.yc is None:
            print("No se pueden visualizar los resultados sin coordenadas.")
            return

        plt.figure(figsize=(8, 8))
        plt.scatter(self.xc, self.yc, s=100, color='lightblue', label='Clientes')
        plt.scatter(self.xc[self.depot_node], self.yc[self.depot_node], s=200, color='red', marker='s', label='Depósito')

        for i in range(self.n):
            plt.annotate(f"N{i}", (self.xc[i] + 0.1, self.yc[i] + 0.1))

        colors = plt.cm.rainbow(np.linspace(0, 1, self.K))
        solution_vars = result.variables_dict
        
        for k in range(self.K):
            for i in range(self.n):
                for j in range(self.n):
                    if solution_vars.get(f"x_{i}_{j}_{k}", 0) > 0.5:
                        plt.plot([self.xc[i], self.xc[j]], [self.yc[i], self.yc[j]], color=colors[k])

        plt.title(f"Solución Cuántica del VRP (Costo: {result.fval:.2f})")
        plt.xlabel("Coordenada X")
        plt.ylabel("Coordenada Y")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    N_CLIENTS = 2
    N_VEHICLES = 2

    vrp_solver = VehicleRoutingQuantumSolver(n_clients=N_CLIENTS, n_vehicles=N_VEHICLES)
    vrp_qp = vrp_solver.build_vrp_qp()
    
    print("Modelo QuadraticProgram construido:")
    print(f"  - Variables: {vrp_qp.get_num_vars()}")
    print(f"  - Restricciones: {len(vrp_qp.linear_constraints)}")
    
    quantum_result = vrp_solver.solve(vrp_qp)
    
    print("\n--- Resultados ---")
    print(f"Solución encontrada: {quantum_result.prettyprint()}")
    
    vrp_solver.visualize_solution(quantum_result)