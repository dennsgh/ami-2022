import React from "react";
import "./pages.css";

// core components
import GridItem from "./Style_components/Grid/GridItem.js";
import GridContainer from "./Style_components/Grid/GridContainer.js";
import Card from "./Style_components/Card/Card.js";

import { Container } from "@material-ui/core";
import StrongerBild from "../images_website/StrongerBild.png";
import SelfsupervisedBild from "../images_website/SelfsupervisedBild.png";
import GIFActive from "../images_website/Self_sup.gif";

import BuchBild from "../images_website/BuchBild.jpg";

import knowledgetransfer from "../images_website/trans_lear.png";
import selfsup from "../images_website/selfsup_bild.png";
import important_samples from "../images_website/Active_Learning.png";
import TransferLearning from "../images_website/TransferLearning.png";

import imageAugment from "../images_website/image_augmentation.png";
import PolicyAugment from "../images_website/augmentation.PNG";
import selfAugment from "../images_website/images_ssl.png";
import RemoveGIF from "../images_website/RemoveGIF.gif";
import Synthese from "../images_website/Synthese.png";

export default function About() {
  return (
    <>
      <div class="textdiv5 padding   paddingVert">
        <GridContainer>
          <GridItem xs={6} sm={6} md={6}>
            <br></br>
            <Container>
              <div class="textHeader">Research</div>
              <br></br>
              <div class="textParaLight">
                {" "}
                <ul>
                  {" "}
                  <div class="textParaLight">
                    {" "}
                    We are working on some of the most complex and interesting
                    challenges in car damage classification.
                  </div>
                </ul>
              </div>
            </Container>
          </GridItem>
          <GridItem xs={6} sm={6} md={6}>
            <br></br>
            <img
              class="center"
              width="50%"
              object-fit="contain"
              src={BuchBild}
              title="Team"
              align="center"
              alt="team"
            />
            <div class="textParaSmallCredWhite textcenter">By Pixabay 2022</div>
          </GridItem>
        </GridContainer>
      </div>
      <div class="textdiv6  ">
        <div class="reveal paddingVert">
          <GridContainer>
            <GridItem xs={12} sm={12} md={12}>
              <Container class="padding">
                <div class="textParaPink">Explore our research:</div>
                <div class="paddingResearch ">
                  <GridContainer>
                    <GridItem xs={6} sm={6} md={6}>
                      <ul>
                        <div class="textParaLight ">
                          <a class="link" href="#fewLabels">
                            Training with few labels
                          </a>
                          <li>- Active learning</li>
                          <li>- Self-supervised learning</li>
                          <li>- Transfer learning</li>
                        </div>
                      </ul>
                    </GridItem>
                    <GridItem xs={6} sm={6} md={6}>
                      <ul>
                        <div class="textParaLight ">
                          <a class="link" href="#expdata">
                            Expand your dataset
                          </a>
                          <li>- Image augmentation</li>
                          <li>- Policy based augmentation</li>
                          <li>- Images for self-supervised learning</li>
                        </div>
                      </ul>
                    </GridItem>
                  </GridContainer>
                </div>
              </Container>
            </GridItem>
          </GridContainer>
        </div>
      </div>
      <div class="textdiv2" id="fewLabels">
        <div class="">
          <div class=" padding reveal paddingVert">
            <GridContainer>
              <GridItem xs={12} sm={12} md={12}>
                <br></br>
                <Container>
                  {" "}
                  <div class="textHeader2"> Training with few labels</div>
                  <br></br>
                </Container>
              </GridItem>

              <GridItem xs={4} sm={4} md={4}>
                <Container>
                  <Card>
                    <img
                      class="center"
                      width="100%"
                      object-fit="contain"
                      src={important_samples}
                      title="Team"
                      align="center"
                      alt="team"
                    />
                  </Card>

                  <div class="textParaBold">
                    Selecting important samples using{" "}
                    <a class="link" href="#activelearning">
                      active learning
                    </a>
                  </div>
                  <br></br>
                  <div class="textPara">
                    Active Learning allow us to choose images which have the
                    most impact on our model performance.
                  </div>
                </Container>
              </GridItem>

              <GridItem xs={4} sm={4} md={4}>
                <Container>
                  <Card>
                    <img
                      class="center"
                      width="100%"
                      object-fit="contain"
                      src={selfsup}
                      title="Team"
                      align="center"
                      alt="team"
                    />{" "}
                  </Card>

                  <div class="textParaBold">
                    Using unlabeled data with{" "}
                    <a class="link" href="#selfsup">
                      self-supervised learning
                    </a>
                  </div>
                  <br></br>
                  <div class="textPara">
                    Through the use of unlabeled images for auxiliary self
                    supervised pretraining we generate a better performance on
                    the classification task afterwards.
                  </div>
                </Container>
              </GridItem>
              <GridItem xs={4} sm={4} md={4}>
                <Container>
                  <Card>
                    <img
                      class="center"
                      width="100%"
                      object-fit="contain"
                      src={knowledgetransfer}
                      title="Team"
                      align="center"
                      alt="team"
                    />{" "}
                  </Card>
                  <div class="textParaBold">
                    Reusing knowledge using{" "}
                    <a class="link" href="#transferlearning">
                      transfer learning
                    </a>
                  </div>
                  <br></br>
                  <div class="textPara">
                    With pretrained models we transfer knowledge gained from
                    larger labeled image datasets, achieving faster convergence.
                  </div>
                </Container>
              </GridItem>
            </GridContainer>
          </div>
        </div>
        <div class="textdiv21" id="activelearning">
          <div class=" padding reveal paddingVert">
            <GridContainer>
              <GridItem xs={6} sm={6} md={6}>
                <Container>
                  {" "}
                  <div class="textParaBold"> Active Learning</div>
                  <br></br>
                  <div class="textPara">
                    <ul>
                      Efficient usage of labeled data
                      <br></br> <br></br>
                      Using intelligent sampling methods important images are
                      identified and labeled using an oracle. These methods
                      include uncertainty, margin and entropy sampling, allowing
                      for faster convergence and better sample efficiency.
                    </ul>
                  </div>
                </Container>
              </GridItem>
              <GridItem xs={6} sm={6} md={6}>
                <img
                  class="center"
                  width="70%"
                  object-fit="contain"
                  src={StrongerBild}
                  title="Team"
                  align="center"
                  alt="team"
                />
              </GridItem>
              <br></br>
            </GridContainer>
          </div>
        </div>

        <div class="textdiv23 paddingVert " id="selfsup">
          <div class="reveal">
            <div class=" padding  ">
              <GridContainer>
                <GridItem xs={6} sm={6} md={6}>
                  <img
                    class="center"
                    width="80%"
                    object-fit="contain"
                    src={GIFActive}
                    title="Team"
                    align="center"
                    alt="team"
                  />
                </GridItem>
                <GridItem xs={6} sm={6} md={6}>
                  <Container>
                    {" "}
                    <div class="textParaBold"> Self-Supervised Learning</div>
                    <br></br>
                    <div class="textPara">
                      <ul>
                        Contrastive Loss for learning hidden representation
                        <br></br> <br></br>
                        Using auxiliary pre-training tasks, such as contrastive
                        loss, we are able to Download use unlabeled data to
                        improve the later classification performance. The
                        projected (using t-SNE) latent space illustrates the
                        distinct classes without prior knowledge of the labels.
                      </ul>
                    </div>
                  </Container>
                </GridItem>
              </GridContainer>
            </div>
            <div class=" padding  ">
              <GridContainer>
                <GridItem xs={6} sm={6} md={6}>
                  <Container>
                    {" "}
                    <br></br> <br></br>
                    <div class="textPara">
                      <ul>
                        Pre-training using unlabeled a dataset <br></br>
                        <br></br>
                        Using auxiliary pre-training tasks, such as contrastive
                        loss, we are able to use unlabeled data to improve the
                        later classification performance. The projected (using
                        t-SNE) latent space illustrates the distinct classes
                        without prior knowledge of the labels.
                      </ul>
                    </div>
                  </Container>
                </GridItem>
                <GridItem xs={6} sm={6} md={6}>
                  <img
                    class="center"
                    width="70%"
                    object-fit="contain"
                    src={SelfsupervisedBild}
                    title="Team"
                    align="center"
                    alt="team"
                  />
                </GridItem>

                <br></br>
              </GridContainer>
            </div>
          </div>
        </div>
        <div class="textdiv22" id="transferlearning">
          <div class=" padding reveal paddingVert">
            <GridContainer>
              <GridItem xs={6} sm={6} md={6}>
                <img
                  class="center"
                  width="70%"
                  object-fit="contain"
                  src={TransferLearning}
                  title="Team"
                  align="center"
                  alt="team"
                />
              </GridItem>
              <GridItem xs={6} sm={6} md={6}>
                <Container>
                  {" "}
                  <div class="textParaBold"> Transfer Learning</div>
                  <br></br>
                  <div class="textPara">
                    <ul>
                      Optimal Performance with Transfer Learning <br></br>
                      <br></br>
                      Transfer Learning introduces a very powerful solution to
                      the research challenge at hand. We use Bayesian
                      Optimization to optimize the hyperparameters of the model,
                      which was pre-trained on ImageNet, for optimal
                      performance.
                    </ul>
                  </div>
                </Container>
              </GridItem>
            </GridContainer>
          </div>
        </div>
      </div>
      <div class="textdiv6" id="expdata">
        <div class=" padding reveal paddingVert">
          <GridContainer>
            <GridItem xs={6} sm={6} md={6}>
              <br></br>
              <Container>
                {" "}
                <div class="textHeader2"> Expanding your dataset</div>
                <br></br>
                <div class="textPara">
                  <ul>
                    In order to apply modern deep learning techniques to a small
                    dataset, we utilise several techniques for artificially
                    increasing the size of the dataset.
                  </ul>
                </div>
              </Container>
            </GridItem>
            <GridItem xs={6} sm={6} md={6}></GridItem>

            <GridItem xs={4} sm={4} md={4}>
              <Container>
                <Card>
                  <img
                    class="center"
                    width="100%"
                    object-fit="contain"
                    src={imageAugment}
                    title="Team"
                    align="center"
                    alt="team"
                  />
                </Card>

                <div class="textParaBold">Image augmentation</div>
                <br></br>
                <div class="textPara">
                  By using hand designed augmentation techniques, we can create
                  artificially modified versions of the input vectors, in order
                  to capture the variation we expect to see at test time.
                </div>
              </Container>
            </GridItem>
            <GridItem xs={4} sm={4} md={4}>
              <Container>
                <Card>
                  <img
                    class="center"
                    width="100%"
                    object-fit="contain"
                    src={PolicyAugment}
                    title="Team"
                    align="center"
                    alt="team"
                  />{" "}
                </Card>
                <div class="textParaBold">
                  Policy based augmentation techniques
                </div>
                <br></br>
                <div class="textPara">
                  Instead of handpicking augmentation techniques, we can also
                  apply automated augmentation techniques to improve
                  performance.
                </div>
              </Container>
            </GridItem>
            <GridItem xs={4} sm={4} md={4}>
              <Container>
                <Card>
                  <img
                    class="center"
                    width="100%"
                    object-fit="contain"
                    src={selfAugment}
                    title="Team"
                    align="center"
                    alt="team"
                  />{" "}
                </Card>

                <div class="textParaBold">
                  Images for self-supervised learning
                </div>
                <br></br>
                <div class="textPara">
                  In addition to the sparse damage classes, we can exploit the
                  large unlabeled dataset for auxiliary pre-training tasks.
                </div>
              </Container>
            </GridItem>
          </GridContainer>
        </div>
      </div>
      <div class="textdiv61">
        <div class="textdiv61" id="activelearning">
          <div class=" padding reveal paddingVert">
            <GridContainer>
              <GridItem xs={6} sm={6} md={6}>
                <Container>
                  {" "}
                  <div class="textParaBold"> Image Augmentation</div>
                  <br></br>
                  <div class="textPara">
                    <ul>
                      Synthetically increase our dataset
                      <br></br> <br></br>
                      Using various methods, realistic augmentation of the
                      training images can be generated. These images assist in
                      making the model become more robust.
                    </ul>
                  </div>
                </Container>
              </GridItem>
              <GridItem xs={6} sm={6} md={6}>
                <img
                  class="center"
                  width="80%"
                  object-fit="contain"
                  src={Synthese}
                  title="Team"
                  align="center"
                  alt="team"
                />
              </GridItem>
              <br></br>
            </GridContainer>
          </div>
        </div>
        <div class="textdiv62" id="selfsup">
          <div class=" padding reveal paddingVert">
            <GridContainer>
              <GridItem xs={6} sm={6} md={6}>
                <img
                  class="center"
                  width="90%"
                  object-fit="contain"
                  src={RemoveGIF}
                  title="Team"
                  align="center"
                  alt="team"
                />
              </GridItem>
              <GridItem xs={6} sm={6} md={6}>
                <Container>
                  {" "}
                  <div class="textParaBold">
                    {" "}
                    Images for Self-Supervised Learning
                  </div>
                  <br></br>
                  <div class="textPara">
                    <ul>
                      Remove non expressive images from dataset
                      <br></br> <br></br>
                      In order to extract meaningful images from the remaining
                      dataset elements, it is necessary to remove static images
                      first. This involves comparing images against each other
                      and removing images that exhibit excessive similarity (
                      euclidean distance, cross-correlation) with the static
                      background.
                    </ul>
                  </div>
                </Container>
              </GridItem>
            </GridContainer>
          </div>
        </div>
      </div>
    </>
  );
}
