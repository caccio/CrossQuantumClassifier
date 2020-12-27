namespace Cazzella.Quantum.CrossQuantum {
    open Microsoft.Quantum.Convert;
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Arrays;
    open Microsoft.Quantum.Math;
    open Microsoft.Quantum.Diagnostics;

    newtype RotationBlock = (
        // QubitIndex and ParameterIndex are 0-based
        fixedQubitIndex: Int,
        rotationQubitIndex: Int,
        rotationParameterIndex: Int,
        rotationAxis: Int
    );

    newtype CircuitModel = (
        network: RotationBlock[],
        measureQubitIndex: Int,
        // measureParameterIndex: Int,
        biasParameterIndex: Int
    );

    function decodeAxis(axisIndex : Int) : Pauli {
        if(axisIndex == 1) {
            return PauliX;
        }
        if(axisIndex == 2) {
            return PauliY;
        }
        if(axisIndex == 3) {
            return PauliZ;
        }
        return PauliI;
    }

    operation ApplyCircuit(circuit : CircuitModel, parameters: Double[], reg: Qubit[]) : Unit is Ctl + Adj {
        for (un in circuit::network) {
            within {
                Controlled X([reg[un::rotationQubitIndex]], reg[un::fixedQubitIndex]);
            }
            apply {
                R(decodeAxis(un::rotationAxis), parameters[un::rotationParameterIndex], reg[un::rotationQubitIndex]);
            }
        }
        // Ry(parameters[circuit::measureParameterIndex], reg[circuit::measureQubitIndex]);
    }

    operation CircuitParameterDerivative(circuit : CircuitModel, parameters : Double[], rotations : Double[][], parameterIndex : Int) : Int {
        let regSize = Length(rotations);
        mutable result = 0;
        mutable shiftedParams = parameters;
        set shiftedParams w/= parameterIndex <- parameters[parameterIndex] + PI();
        using ((reg, a) = (Qubit[regSize], Qubit())) {
            EncodeStatus(rotations, reg);
            Reset(a);
            H(a);
            Controlled ApplyCircuit([a], (circuit, parameters, reg));
            Controlled Z([a], reg[circuit::measureQubitIndex]);
            X(a);
            Controlled ApplyCircuit([a], (circuit, shiftedParams, reg));
            X(a);
            H(a);
            if (M(a) == One) { set result = 1; }
            ResetAll(reg);
            Reset(a);
            return result;
        }
    }

    operation Classify(circuit : CircuitModel, parameters : Double[], rotations : Double[][]) : Int {
        let regSize = Length(rotations);
        let n = regSize;
        mutable result = 0;
        using (reg = Qubit[regSize]) {
            EncodeStatus(rotations, reg);
            ApplyCircuit(circuit, parameters, reg);
            if (M(reg[circuit::measureQubitIndex]) == One) { set result = 1; }
            ResetAll(reg);
            return result;
        }
    }

    operation EncodeStatus(rotations : Double[][], register : Qubit[]) : Unit {
        let n = Length(register);
        for (s in n..-1..1) {
            for (j in 2^(n-s)..-1..1) {
                if (s==n) {                  
                    Ry(rotations[s-1][j-1], register[s-1]);
                } else {
                    within {
                        for (k in s+1..1..n) {
                            if ((2^(n-k) &&& (j-1)) == 0) {
                                X(register[n-k+s]);
                            }
                        }
                    }
                    apply {
                        Controlled Ry(register[s..n-1], (rotations[s-1][j-1], register[s-1]));
                    }
                }
            }
        }
    }
}