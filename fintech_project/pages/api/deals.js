const deals = {
    foodie: [{
        title: "$10 off SaladStop!",
        message: "Your next lunch is on us! Grab this deal and receive $10 off your purchase at SaladStop.",
        imageUrl: "/images/merchants/salad-stop.png",
        personaImageUrl: "/images/personas/foodie.png"
    }],
    traveller: [{
        title: "3% off Singapore Airlines!",
        message: "Your next adventure is on us! Grab this deal and receive $3% off your purchase at Singapore Airlines.",
        imageUrl: "/images/merchants/sia.png",
        personaImageUrl: "/images/personas/traveller.png"
    }],
    fashion: [{
        title: "$10 off Valentino!",
        message: "Your next purchase is on us! Grab this deal and receive $10 off your purchase at Valentino.",
        imageUrl: "/images/merchants/valentino.png",
        personaImageUrl: "/images/personas/fashion.png"
    }]
};

export default (req, res) => {
    if (req.method === "GET") {
        console.log(req.query);
        const persona = req.query.persona;
        let possibleDeals = [];

        if (persona === "Foodie") {
            possibleDeals = deals.foodie;
        } else if (persona === "Traveller") {
            possibleDeals = deals.traveller;
        } else {
            possibleDeals = deals.fashion;
        }

        // get a random deal
        if (possibleDeals.length > 0) {
            const randNum = Math.random() * possibleDeals.length;
            res.statusCode = 200;
            res.end(
                JSON.stringify({
                    deal: possibleDeals[0]
                })
            );
        } else {
            res.statusCode = 404;
            res.end();
        }
    }
};