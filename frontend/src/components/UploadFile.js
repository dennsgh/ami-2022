// @material-ui/core components
import "./pages.css";
import React from "react";
import ImageUploading from "react-images-uploading";

// core components
import GridItem from "./Style_components/Grid/GridItem.js";
import GridContainer from "./Style_components/Grid/GridContainer.js";
import { Container } from "@material-ui/core";
import ImageWrapper from "./ImageWrapper.js";

import Headlamp from "../images_website/headlamp.jpg";
import Brain from "../images_website/brain.jpg";

// export default function TableList() {
function UploadFile({
  maxNumber,
  mode,
  images,
  setImages,
  predictions,
  setPredictions,
  imageLabels,
  setImageLabels,
}) {
  // Todo: const currentImage = [];
  // const [annotations, setAnnotations] = useState([]);
  const onChange = (imageList, addUpdateIndex) => {
    setImages(imageList);
  };

  return (
    <div>
      <div class="textdiv5 padding paddingtop">
        <GridContainer>
          {mode === "Labeling" && (
            <>
              <GridItem xs={6} sm={6} md={6}>
                <br></br>
                <Container>
                  <div class="textHeader">Labeling</div>
                  <br></br>
                  <div class="textParaLightSmall">
                    {" "}
                    <ul>
                      Improve our model here: Upload your pictures and select
                      one.
                      <div class="break">
                        <br></br>
                        <br></br>
                      </div>
                      Choose a label by using the LABEL button.
                      <div class="break">
                        <br></br>
                        <br></br>
                      </div>
                      In the case you want to experience our classification
                      model, simply move over to the PREDICTION subpage.
                    </ul>
                  </div>
                </Container>
              </GridItem>
              <GridItem xs={6} sm={6} md={6}>
                <br></br>
                <img
                  class="center "
                  width="70%"
                  object-fit="contain"
                  src={Headlamp}
                  title="Team"
                  align="center"
                  alt="team"
                />
                <div class="textParaSmallCredWhite textcenter">
                  By Pixabay 2022
                </div>
              </GridItem>
            </>
          )}
          {mode === "Prediction" && (
            <>
              <GridItem xs={6} sm={6} md={6}>
                <br></br>
                <Container>
                  <div class="textHeader">Prediction</div>
                  <br></br>
                  <div class="textParaLightSmall">
                    {" "}
                    <ul>
                      Experience our classification model here: Upload your
                      pictures and select one.
                      <div class="break">
                        <br></br>
                        <br></br>
                      </div>
                      Press PREDICT and let us classify your damage!
                      <div class="break">
                        <br></br>
                        <br></br>
                      </div>
                      In the case you want to relabel it, simply move over to
                      the LABELING subpage.
                    </ul>
                  </div>
                </Container>
              </GridItem>
              <GridItem xs={6} sm={6} md={6}>
                <br></br>
                <img
                  class="center"
                  width="60%"
                  object-fit="contain"
                  src={Brain}
                  title="Team"
                  align="center"
                  alt="team"
                />
                <div class="textParaSmallCred textcenter">By Pixabay 2022</div>
              </GridItem>
            </>
          )}
        </GridContainer>
      </div>

      <div class="textdiv5  ">
        <GridContainer>
          <GridItem xs={12} sm={12} md={12}>
            <ImageUploading
              multiple
              value={images}
              onChange={onChange}
              maxNumber={maxNumber}
              dataURLKey="data_url"
            >
              {({
                imageList,
                onImageUpload,
                onImageRemoveAll,
                onImageUpdate,
                onImageRemove,
                isDragging,
                dragProps,
              }) => (
                <ImageWrapper
                  isDragging={isDragging}
                  dragProps={dragProps}
                  onImageRemoveAll={onImageRemoveAll}
                  onImageUpdate={onImageUpdate}
                  onImageRemove={onImageRemove}
                  mode={mode}
                  images={images}
                  onImageUpload={onImageUpload}
                  predictions={predictions}
                  setPredictions={setPredictions}
                  imageLabels={imageLabels}
                  setImageLabels={setImageLabels}
                ></ImageWrapper>
              )}
            </ImageUploading>
          </GridItem>
        </GridContainer>
      </div>
    </div>
  );
}
export default UploadFile;
