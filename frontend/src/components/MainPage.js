import React from "react";
import "./pages.css";

// @material-ui/core
import { Container } from "@material-ui/core";

// core components
import GridItem from "./Style_components/Grid/GridItem.js";
import GridContainer from "./Style_components/Grid/GridContainer.js";

import MainCar from "../images_website/MainCar.jpg";
import AccuracyAll from "../images_website/Accuracy-all.png";

import Deepold from "../images_website/DeepoldTrans.png";
import GIFActive from "../images_website/Self_sup.gif";

export default function Dashboard() {
  return (
    <>
      <div class="textdiv5 padding paddingVert ">
        {" "}
        <GridContainer>
          <GridItem xs={12} sm={12} md={6} lg={6} xl={6}>
            <br></br>

            <Container>
              <ul>
                {" "}
                <div class="textParaLight  padding">
                  {" "}
                  AlphaDamage is an AI system developed by Deep.old that makes
                  classifying car damages simple and accurate for everyone.
                  <br></br> Deep.old is your partner when working with a car
                  damage classification task.
                  <br></br> Try it yourself!{" "}
                </div>
              </ul>
            </Container>
          </GridItem>
          <GridItem xs={12} sm={12} md={6} lg={6} xl={6}>
            <div class="iframe-behaelter-behaelter">
              <div class="iframe-behaelter">
                <iframe
                  width="560"
                  height="315"
                  src="https://www.youtube.com/embed/mnsGymMDK68"
                  title="YouTube video player"
                  frameborder="0"
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                  allowfullscreen
                ></iframe>
              </div>
            </div>
          </GridItem>
        </GridContainer>
        <div class="padding reveal paddingVert ">
          <GridContainer>
            <GridItem xs={6} sm={6} md={6} lg={6} xl={6}>
              <img
                width="100%"
                object-fit="contain"
                src={MainCar}
                title="Team"
                align="center"
                alt="team"
              />
              <div class="textParaSmallCred ">By Pixabay 2022</div>
            </GridItem>

            <GridItem xs={6} sm={6} md={6} lg={6} xl={6}>
              <div class="centeralign">
                <img
                  class="center"
                  align="center"
                  margin="10vw"
                  width="70%"
                  src={Deepold}
                  alt="logo"
                />
              </div>
            </GridItem>
          </GridContainer>
        </div>
      </div>
      <div class="textdiv6 padding ">
        <div class="reveal paddingVert">
          <GridContainer>
            <GridItem xs={6} sm={6} md={6} lg={6} xl={6}>
              <br></br>
              <Container>
                {" "}
                <div class="textHeader"> Cutting Edge Technologies</div>
                <div class="textPara ">
                  <ul>
                    We combine cutting edge technologies like data augmentation,
                    contrastive loss and diverse learning methods like transfer
                    learning, self-supervised learning or active learning in one
                    product.
                  </ul>
                </div>
              </Container>
            </GridItem>
            <GridItem xs={6} sm={6} md={6} lg={6} xl={6}>
              <img
                width="90%"
                object-fit="contain"
                src={AccuracyAll}
                title="Team"
                align="center"
                alt="team"
              />

              <div class="textParaSmall ">
                We are comparing the accuracy of three different models.
              </div>
            </GridItem>
          </GridContainer>
        </div>
      </div>

      <div class="textdiv2 padding ">
        <div class="reveal paddingVert">
          <GridContainer>
            <GridItem xs={6} sm={6} md={6} lg={6} xl={6}>
              <img
                class="center"
                width="90%"
                object-fit="contain"
                src={GIFActive}
                title="Team"
                align="center"
                alt="team"
              />
            </GridItem>
            <GridItem xs={6} sm={6} md={6} lg={6} xl={6}>
              <br></br>
              <Container>
                <div class="textHeader">State of the Art Techniques </div>

                <div class="textPara ">
                  <ul>
                    We deploy state of the art and sample efficient techniques
                    achieving the best results for our customers.
                  </ul>
                </div>
              </Container>
            </GridItem>
          </GridContainer>
        </div>
      </div>
    </>
  );
}
