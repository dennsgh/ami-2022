import React from "react";
import "./pages.css";
// @material-ui/core
import { makeStyles } from "@material-ui/core/styles";

// core components
import GridItem from "./Style_components/Grid/GridItem.js";
import GridContainer from "./Style_components/Grid/GridContainer.js";
import Card from "./Style_components/Card/Card.js";
import CardHeader from "./Style_components/Card/CardHeader.js";
import CardIcon from "./Style_components/Card/CardIcon.js";
import CardFooter from "./Style_components/Card/CardFooter.js";

import Team from "../images_website/team_TUM.png";
import Nils from "../images_website/Nils.jpg";
import Anouar from "../images_website/Anouar.jpg";
import Amadou from "../images_website/Amadou.jpg";
import Melina from "../images_website/Melina.png";
import Linyan from "../images_website/Linyan.png";
import Gregor from "../images_website/Gregor.png";
import Xiagang from "../images_website/Xiagang.jpg";
import Dennis from "../images_website/Dennis.png";
import Maggi from "../images_website/maggi.jpg";

import styles from "./Style_components/jss/material-dashboard-react/views/dashboardStyle.js";
import { Container } from "@material-ui/core";

const useStyles = makeStyles(styles);

export default function Dashboard() {
  const classes = useStyles();
  return (
    <>
      <div class="textdiv5 padding  paddingVert">
        <GridContainer>
          <GridItem xs={12} sm={12} md={6}>
            <br></br>
            <Container>
              <div class="textHeader">Team</div>
              <br></br>
              <div class="textParaLight">
                {" "}
                <ul>
                  Deep.old has worked together for 5 month now and created
                  AlphaDamage. The people behind it contribute with their
                  experience in data sciene, software architecture and UX/UI to
                  AlphaDamage. Get a closer look at the Deep.old team here:
                </ul>
              </div>
            </Container>
          </GridItem>
          <GridItem xs={12} sm={12} md={6}>
            <br></br>

            <img
              width="90%"
              object-fit="contain"
              src={Team}
              title="Team"
              align="center"
              alt="team"
            />

            <div class="textParaSmall">
              Nils, Dennis, Amadou, Magdalena, Linyan, Xiaogang, Anouar, Melina
              and Gregor <br></br>(starting left, top row)
            </div>
          </GridItem>
        </GridContainer>
      </div>

      <div class="textdiv6 padding paddingVert ">
        <div class="reveal">
          <GridContainer>
            <GridItem xs={12} sm={4} md={4}>
              <Card>
                <CardHeader color="info" stats icon>
                  <CardIcon color="info">
                    <div>
                      <img
                        height="100px"
                        width="100px"
                        align="left"
                        src={Linyan}
                        alt="team"
                      />
                    </div>
                  </CardIcon>
                  <p
                    align="left"
                    className={classes.cardCategory}
                    style={{
                      color: "#000000",
                      fontSize: "1.2vw",
                      height: "2vw",
                      fontFamily: "Titillium Web",
                      fontWeight: "lighter",
                    }}
                  >
                    Linyan
                  </p>
                  <br></br>
                  <div class="textPosition">Modeling - Research</div>
                </CardHeader>
                <CardFooter stats>
                  <div
                    className={classes.stats}
                    style={{
                      color: "#000000",
                      fontSize: "1.2vw",
                      fontFamily: "Titillium Web",
                      fontWeight: "lighter",
                    }}
                  >
                    I am enthusiastic about discovering and applying new
                    technical solutions.
                  </div>
                </CardFooter>
              </Card>
            </GridItem>
            <GridItem xs={12} sm={4} md={4}>
              <Card>
                <CardHeader color="info" stats icon>
                  <CardIcon color="info">
                    <div>
                      <img
                        height="100px"
                        width="100px"
                        align="right"
                        src={Amadou}
                        alt="team"
                      />
                    </div>
                  </CardIcon>
                  <p
                    align="left"
                    className={classes.cardCategory}
                    style={{
                      color: "#000000",
                      fontSize: "1.2vw",
                      height: "2vw",
                      fontFamily: "Titillium Web",
                      fontWeight: "lighter",
                    }}
                  >
                    Amadou
                  </p>
                  <br></br>
                  <div class="textPosition">Modeling - Backend </div>
                </CardHeader>
                <CardFooter stats>
                  <div
                    className={classes.stats}
                    style={{
                      color: "#000000",
                      fontSize: "1.2vw",
                      fontFamily: "Titillium Web",
                      fontWeight: "lighter",
                    }}
                  >
                    Passionate about bring the optimal solution to our code
                    base.
                  </div>
                </CardFooter>
              </Card>
            </GridItem>

            <GridItem xs={12} sm={4} md={4}>
              <Card>
                <CardHeader color="info" stats icon>
                  <CardIcon color="info">
                    <div>
                      <img
                        height="100px"
                        width="100px"
                        align="left"
                        src={Melina}
                        alt="team"
                      />
                    </div>
                  </CardIcon>
                  <p
                    align="left"
                    className={classes.cardCategory}
                    style={{
                      fontSize: "1.2vw",
                      color: "#000000",
                      height: "2vw",
                      fontFamily: "Titillium Web",
                      fontWeight: "lighter",
                    }}
                  >
                    Melina
                  </p>
                  <br></br>
                  <div class="textPosition">Modeling - Organization</div>
                </CardHeader>
                <CardFooter stats>
                  <div
                    className={classes.stats}
                    style={{
                      color: "#000000",
                      fontSize: "1.2vw",
                      fontFamily: "Titillium Web",
                      fontWeight: "lighter",
                    }}
                  >
                    I like learning more about new technologies while discussing
                    and exploring their limits with my team.
                  </div>
                </CardFooter>
              </Card>
            </GridItem>
          </GridContainer>
        </div>
        <div class="reveal ">
          <GridContainer>
            <GridItem xs={12} sm={4} md={4}>
              <Card>
                <CardHeader color="info" stats icon>
                  <CardIcon color="info">
                    <div>
                      <img
                        height="100px"
                        width="100px"
                        align="right"
                        src={Gregor}
                        alt="team"
                      />
                    </div>
                  </CardIcon>
                  <p
                    align="left"
                    className={classes.cardCategory}
                    style={{
                      color: "#000000",
                      fontSize: "1.2vw",
                      height: "2vw",
                      fontFamily: "Titillium Web",
                      fontWeight: "lighter",
                    }}
                  >
                    Gregor
                  </p>
                  <br></br>
                  <div class="textPosition">
                    Technical Set-up - Preprocessing
                  </div>
                </CardHeader>
                <CardFooter stats>
                  <div
                    className={classes.stats}
                    style={{
                      color: "#000000",
                      fontSize: "1.2vw",
                      fontFamily: "Titillium Web",
                      fontWeight: "lighter",
                    }}
                  >
                    I love working on innovative software solutions to make life
                    easier and better.
                  </div>
                </CardFooter>
              </Card>
            </GridItem>
            <GridItem xs={12} sm={4} md={4}>
              <Card>
                <CardHeader color="info" stats icon>
                  <CardIcon color="info">
                    <div>
                      <img
                        height="100px"
                        width="100px"
                        align="left"
                        src={Anouar}
                        alt="team"
                      />
                    </div>
                  </CardIcon>
                  <p
                    align="left"
                    className={classes.cardCategory}
                    style={{
                      color: "#000000",
                      fontSize: "1.2vw",
                      height: "2vw",
                      fontFamily: "Titillium Web",
                      fontWeight: "lighter",
                    }}
                  >
                    Anouar
                  </p>
                  <br></br>
                  <div class="textPosition">Allrounder - Backend</div>
                </CardHeader>
                <CardFooter stats>
                  <div
                    className={classes.stats}
                    style={{
                      color: "#000000",
                      fontSize: "1.2vw",
                      fontFamily: "Titillium Web",
                      fontWeight: "lighter",
                    }}
                  >
                    I'm just here for the challenge.
                  </div>
                </CardFooter>
              </Card>
            </GridItem>

            <GridItem xs={12} sm={4} md={4}>
              <Card>
                <CardHeader color="info" stats icon>
                  <CardIcon color="info">
                    <div>
                      <img
                        height="100px"
                        width="100px"
                        align="left"
                        src={Nils}
                        alt="team"
                      />
                    </div>
                  </CardIcon>
                  <p
                    align="left"
                    className={classes.cardCategory}
                    style={{
                      color: "#000000",
                      fontSize: "1.2vw",
                      height: "2vw",
                      fontFamily: "Titillium Web",
                      fontWeight: "lighter",
                    }}
                  >
                    Nils
                  </p>
                  <br></br>
                  <div class="textPosition">Technical Set-up - Deployment</div>
                </CardHeader>
                <CardFooter stats>
                  <div
                    className={classes.stats}
                    style={{
                      color: "#000000",
                      fontSize: "1.2vw",
                      fontFamily: "Titillium Web",
                      fontWeight: "lighter",
                    }}
                  >
                    Imagine if we could predict the future or maybe even
                    classify it.
                  </div>
                </CardFooter>
              </Card>
            </GridItem>
          </GridContainer>
        </div>
        <div class="reveal">
          <GridContainer>
            <GridItem xs={12} sm={4} md={4}>
              <Card>
                <CardHeader color="info" stats icon>
                  <CardIcon color="info">
                    <div>
                      <img
                        height="100px"
                        width="100px"
                        align="left"
                        src={Xiagang}
                        alt="team"
                      />
                    </div>
                  </CardIcon>
                  <p
                    align="left"
                    className={classes.cardCategory}
                    style={{
                      color: "#000000",
                      fontSize: "1.2vw",
                      height: "2vw",
                      fontFamily: "Titillium Web",
                      fontWeight: "lighter",
                    }}
                  >
                    Xiaogang
                  </p>
                  <br></br>
                  <div class="textPosition">Photography - Preprocessing</div>
                </CardHeader>
                <CardFooter stats>
                  <div
                    className={classes.stats}
                    style={{
                      color: "#000000",
                      fontSize: "1.2vw",
                      fontFamily: "Titillium Web",
                      fontWeight: "lighter",
                    }}
                  >
                    I believe that intelligence creates a more beautiful life.
                  </div>
                </CardFooter>
              </Card>
            </GridItem>
            <GridItem xs={12} sm={4} md={4}>
              <Card>
                <CardHeader color="info" stats icon>
                  <CardIcon color="info">
                    <div>
                      <img
                        height="100px"
                        width="100px"
                        align="left"
                        src={Maggi}
                        alt="team"
                      />
                    </div>
                  </CardIcon>
                  <p
                    align="left"
                    className={classes.cardCategory}
                    style={{
                      color: "#000000",
                      fontSize: "1.2vw",
                      height: "2vw",
                      fontFamily: "Titillium Web",
                      fontWeight: "lighter",
                    }}
                  >
                    Magdalena
                  </p>
                  <br></br>
                  <div class="textPosition">
                    Frontend Design - Preprocessing
                  </div>
                </CardHeader>
                <CardFooter stats>
                  <div
                    className={classes.stats}
                    style={{
                      color: "#000000",
                      fontSize: "1.2vw",
                      fontFamily: "Titillium Web",
                      fontWeight: "lighter",
                    }}
                  >
                    I am passionate about coding and developing new solutions
                    for a better future.
                  </div>
                </CardFooter>
              </Card>
            </GridItem>
            <GridItem xs={12} sm={4} md={4}>
              <Card>
                <CardHeader color="info" stats icon>
                  <CardIcon color="info">
                    <div>
                      <img
                        height="100px"
                        width="100px"
                        align="right"
                        src={Dennis}
                        alt="team"
                      />
                    </div>
                  </CardIcon>
                  <p
                    align="left"
                    className={classes.cardCategory}
                    style={{
                      color: "#000000",
                      fontSize: "1.2vw",
                      height: "2vw",
                      fontFamily: "Titillium Web",
                      fontWeight: "lighter",
                    }}
                  >
                    Dennis
                  </p>
                  <br></br>
                  <div class="textPosition">Media - Preprocessing</div>
                </CardHeader>
                <CardFooter stats>
                  <div
                    className={classes.stats}
                    style={{
                      color: "#000000",
                      fontSize: "1.2vw",
                      fontFamily: "Titillium Web",
                      fontWeight: "lighter",
                    }}
                  >
                    My parents said either doctor or engineer, so I decided to
                    disappoint them.
                  </div>
                </CardFooter>
              </Card>
            </GridItem>
          </GridContainer>
        </div>
      </div>
    </>
  );
}
