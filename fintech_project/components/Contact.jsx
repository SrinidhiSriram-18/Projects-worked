const Contact = () => (
  <div className="container contact-form">
    <div className="row">
      <div className="col-sm-8 offset-md-2">
        <form action="https://formspree.io/harishv207@gmail.com" method="POST">
          <div className="form-group">
            <label>Name</label>
            <input
              name="name"
              type="text"
              className="form-control"
              placeholder="John"
            />
          </div>
          <div className="form-group">
            <label>Email address</label>
            <input
              name="email"
              type="email"
              className="form-control"
              placeholder="name@example.com"
            />
          </div>
          <div className="form-group">
            <label>Comments / Feedback</label>
            <textarea name="comments" className="form-control" rows="3" />
          </div>
          <input type="submit" className="btn btn-primary" value="Submit" />
        </form>
      </div>
    </div>
  </div>
);

export default Contact;
