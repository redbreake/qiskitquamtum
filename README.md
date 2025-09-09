# Qiskit Quantum VRP Solver

Este proyecto implementa una solución al Problema de Enrutamiento de Vehículos (VRP) utilizando la biblioteca de computación cuántica Qiskit. El código está diseñado para ser compatible con Qiskit 1.0+ (incluyendo 2.x) y demuestra cómo formular problemas de optimización combinatoria para ser resueltos con algoritmos cuánticos, específicamente utilizando `SamplingVQE` y `StatevectorSampler`.

## Características

- **Generación de Instancias VRP:** Genera instancias aleatorias del Problema de Enrutamiento de Vehículos.
- **Formulación Cuántica:** Transforma el VRP en un `QuadraticProgram` compatible con los algoritmos de optimización cuántica de Qiskit.
- **Resolución Cuántica:** Utiliza `SamplingVQE` con `StatevectorSampler` y `RealAmplitudes` como ansatz para encontrar soluciones aproximadas.
- **Visualización de Resultados:** Muestra gráficamente las rutas encontradas por los vehículos.

## Requisitos

Asegúrate de tener Python 3.8+ instalado. Puedes instalar las dependencias necesarias usando `pip`:

```bash
pip install qiskit qiskit-optimization matplotlib numpy
```

## Uso

1.  **Clonar el Repositorio:**
    ```bash
    git clone https://github.com/redbreake/qiskitquamtum.git
    cd qiskitquamtum
    ```

2.  **Ejecutar el Solucionador VRP:**
    El archivo `d.py` contiene la implementación principal. Puedes ejecutarlo directamente:
    ```bash
    python d.py
    ```
    Este script configurará un VRP con un número predefinido de clientes y vehículos, lo resolverá usando el enfoque cuántico y visualizará la solución.

## Estructura del Proyecto

-   `d.py`: Contiene la clase `VehicleRoutingQuantumSolver` que encapsula la lógica para generar instancias VRP, construir el `QuadraticProgram`, resolverlo cuánticamente y visualizar los resultados.

## Contribuciones

Las contribuciones son bienvenidas. Si tienes sugerencias o mejoras, no dudes en abrir un *issue* o enviar un *pull request*.

## Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.