const offers = {
  "Singapore Airlines": [
    {
      title: "Transport to Airport",
      message: "Need a ride from your home to airport? Leave the hassle to us!",
      imageUrl: "/images/app/car.png"
    },
    {
      title: "Travel insurance",
      message:
        "Are you looking for a travel insurance for your next adventure?",
      imageUrl: "/images/app/insurance.png"
    }
  ],
  KFC: [
    {
      title: "Get a dessert to go with your meal!",
      message: "Savour that McFlurry for just $1.20 now!",
      imageUrl: "/images/app/ice-cream.png"
    },
    {
      title: "Get a coffee to go!",
      message:
        "Check out Starbucks for this instant 20% discount on any latte you order!",
      imageUrl: "/images/app/coffee.png"
    }
  ],
  Zalora: [
    {
      title: "Fancy the latest fashion?",
      message: "Shop at A&F with this $10 voucher. Just for you!",
      imageUrl: "/images/app/t-shirt.png"
    },
    {
      title: "Run for your livessss!",
      message:
        "Get the latest Nike shoes at a 20% discount. Only valid on Sundays!",
      imageUrl: "/images/app/shoe.png"
    }
  ]
};

export default (req, res) => {
  if (req.method === "GET") {
    console.log(req.query);
    const merchant = req.query.merchant;
    const possibleOffers = offers[merchant];
    console.log(merchant);

    res.end(JSON.stringify(possibleOffers));
  } else {
    res.statusCode = 404;
    res.end();
  }
};
