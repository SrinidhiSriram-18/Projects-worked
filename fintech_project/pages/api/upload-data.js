const brain = require("./brain");

export default (req, res) => {
    if (req.method === "POST") {
        res.statusCode = 200;

        const result = brain.getPersona(req.body);
        res.end(
            JSON.stringify({
                message: "Thanks for uploading your data. You seem to be a " +
                    result.persona +
                    "!",
                persona: result.persona,
                transactionAmount: result.totalTransactionAmount
            })
        );
    } else {
        // Handle the rest of your HTTP methods
        res.setHeader("Content-Type", "application/json");
        res.statusCode = 200;
        res.end(JSON.stringify({ name: "AIS" }));
    }
};