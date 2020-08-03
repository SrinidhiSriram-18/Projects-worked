import React from "react";
import Link from "next/link";

const Footer = () => (
  <footer id="footer">
    <div className="section container">
      <div className="row">
        <div className="col-sm-12 text-center">
          <span className="footer-links">
            <Link href="/index">
              <a>Home</a>
            </Link>
          </span>
          <span className="footer-links">
            <Link href="/about">
              <a>About</a>
            </Link>
          </span>
          <span className="footer-links">
            <Link href="/contact">
              <a>Contact</a>
            </Link>
          </span>
          <span className="footer-links">
            <Link href="/try">
              <a>Try Now</a>
            </Link>
          </span>
        </div>
      </div>
      <div className="row credit">
        <div className="col-sm-12 text-center">
          © 2019 Actionable Intelligence System Inc. All Rights Reserved.
        </div>
      </div>
    </div>
  </footer>
);
export default Footer;
