function Card() {
  return (
    <div className="credit-card">
      <div className="row">
        <div className="col-sm-3 text-center">
          <div className="card-balance">$10000</div>
        </div>
      </div>
      <div className="cardholder-name">Jackson Maine</div>
      <div className="card-text">Card number</div>
      <div className="card-number">XXXX-XXXX-XXXX-1281</div>
      <div className="row">
        <div className="col-sm-8">
          <div className="card-text">Valid from</div>
          <div className="valid-from">01/19</div>
        </div>
        <div className="col-sm-4">
          <div className="card-text">Valid through</div>
          <div className="valid-through">07/25</div>
        </div>
      </div>
    </div>
  );
}

export default Card;
