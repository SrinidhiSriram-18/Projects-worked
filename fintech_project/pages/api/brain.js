const personaTypes = require("./persona-types");
const merchantMapping = require("./merchant-mapping");

var personaMap = {};

const brain = {
    getPersona: function(data) {
        let totalTransactionAmount = 0;
        data.map(transaction => {
            const merchantName = transaction[1];
            totalTransactionAmount += parseFloat(transaction[2]);
            const transactionType = merchantMapping[merchantName];
            if (transactionType in personaMap) {
                personaMap[transactionType] = personaMap[transactionType] + 1;
            } else {
                personaMap[transactionType] = 1;
            }
        });
        // get highest transaction as persona
        console.log(personaMap);
        const persona = Object.keys(personaMap).reduce((a, b) =>
            personaMap[a] > personaMap[b] ? a : b
        );
        // reset personaMap
        personaMap = {};
        return {
            persona: personaTypes.personaReverseMapping[persona],
            totalTransactionAmount: totalTransactionAmount
        };
    }
};

module.exports = brain;