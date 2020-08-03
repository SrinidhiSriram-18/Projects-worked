const trainingData = [{
        input: {
            foodieSpend: 0.05,
            foodieTxns: 0.005,
            explorerSpend: 0.02,
            explorerTxns: 0.002,
            fashionSpend: 0.03,
            fashionTxns: 0.003
        },
        output: {
            foodie: 1
        }
    },
    {
        input: {
            foodieSpend: 0.01,
            foodieTxns: 0.0002,
            explorerSpend: 0.15,
            explorerTxns: 0.0006,
            fashionSpend: 0.04,
            fashionTxns: 0.0002
        },
        output: {
            explorer: 1
        }
    },
    {
        input: {
            foodieSpend: 0.1,
            foodieTxns: 0.002,
            explorerSpend: 0.1,
            explorerTxns: 0.001,
            fashionSpend: 0.3,
            fashionTxns: 0.002
        },
        output: {
            fashionFiesta: 1
        }
    },
    {
        input: {
            foodieSpend: 0.02,
            foodieTxns: 0.001,
            explorerSpend: 0,
            explorerTxns: 0,
            fashionSpend: 0.2,
            fashionTxns: 0.001
        },
        output: {
            fashionFiesta: 1
        }
    },
    {
        input: {
            foodieSpend: 0.0442,
            foodieTxns: 0.0012,
            explorerSpend: 0.0022,
            explorerTxns: 0.0002,
            fashionSpend: 0,
            fashionTxns: 0
        },
        output: {
            foodie: 1
        }
    },
    {
        input: {
            foodieSpend: 0.02,
            foodieTxns: 0.001,
            explorerSpend: 0.223,
            explorerTxns: 0.0001,
            fashionSpend: 0.012,
            fashionTxns: 0.0022
        },
        output: {
            explorer: 1
        }
    }
];

module.exports = trainingData;