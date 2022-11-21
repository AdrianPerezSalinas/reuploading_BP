from qibo.models import Circuit
from qibo import gates
from qibo import hamiltonians
from qibo.models import VQE
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--Qubits', type=int, default=8, help='max number of qubits')

def main(Qubits):
    Qubits += 1
    h = .5

    fig, ax = plt.subplots()

    vars = []

    for qubits in range(2, Qubits, 2):
        print(qubits)

        ansatz = Circuit(qubits)
        for l in range(qubits):
            for q in range(qubits):
                ansatz.add(gates.RZ(q, theta=0))
                ansatz.add(gates.RY(q, theta=0))
                ansatz.add(gates.RZ(q, theta=0))
            for q in range(0, qubits - 1, 2):
                ansatz.add(gates.CZ(q, q + 1))
            for q in range(qubits):
                ansatz.add(gates.RZ(q, theta=0))
                ansatz.add(gates.RY(q, theta=0))
                ansatz.add(gates.RZ(q, theta=0))
            for q in range(1, qubits - 1, 2):
                ansatz.add(gates.CZ(q, q + 1))
            ansatz.add(gates.CZ(qubits - 1, 0))
        
        for q in range(qubits):
            ansatz.add(gates.RZ(q, theta=0))
            ansatz.add(gates.RY(q, theta=0))
            ansatz.add(gates.RZ(q, theta=0))

        
        ham = hamiltonians.TFIM(qubits, h=h)

        num = len(ansatz.get_parameters())
        samples = 10 * num

        grad = []

        for s in range(samples):
            params = 2 * np.pi * np.random.rand(num)

            params1 = params.copy()
            params1[0] += np.pi / 2

            ansatz.set_parameters(params1)
            e1 = ham.expectation(ansatz.execute())

            params2 = params.copy()
            params2[0] -= np.pi / 2

            ansatz.set_parameters(params2)
            e2 = ham.expectation(ansatz.execute())

            grad.append(.5 * (e2 - e1))

        vars.append(np.var(grad))
        
    ax.scatter(list(range(2, Qubits, 2)), vars, c='k')

    ax.set_yscale('log')
    fig.savefig('RZYZ-ising.pdf')


if __name__ == '__main__':
    args = vars(parser.parse_args())
    main(**args)

