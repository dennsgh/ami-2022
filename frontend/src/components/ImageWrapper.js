import { useEffect, useState } from "react";
import "./pages.css";
// @material-ui/core components
import { makeStyles } from "@material-ui/core/styles";
import { Container } from "@material-ui/core";
import Box from "@mui/material/Box";

import Button from "@material-ui/core/Button";
import ButtonGroup from "@material-ui/core/ButtonGroup";
import { InputLabel } from "@mui/material";
import { Select } from "@mui/material";
import { MenuItem } from "@mui/material";
import { FormControl } from "@mui/material";
import UploadIcon from "@mui/icons-material/Upload";
import TextField from "@material-ui/core/TextField";
import DeleteIcon from "@mui/icons-material/Delete";
import { Slide } from "@mui/material";

import { FormControlLabel } from "@material-ui/core";
import { Checkbox } from "@material-ui/core";
import GridItem from "./Style_components/Grid/GridItem.js";
import GridContainer from "./Style_components/Grid/GridContainer.js";
import Card from "./Style_components/Card/Card.js";
import CardBody from "./Style_components/Card/CardBody.js";
import CardHeader from "./Style_components/Card/CardHeader.js";

const useStyles = makeStyles({
  cardCategoryWhite: {
    "&,& a,& a:hover,& a:focus": {
      color: "rgba(255,255,255,.62)",
      margin: "0",
      fontSize: "14px",
      marginTop: "0",
      marginBottom: "0",
    },
    "& a,& a:hover,& a:focus": {
      color: "#FFFFFF",
    },
  },
  cardTitleWhite: {
    color: "#FFFFFF",
    marginTop: "0px",
    minHeight: "auto",
    fontWeight: "300",
    fontFamily: "'Roboto', 'Helvetica', 'Arial', sans-serif",
    marginBottom: "3px",
    textDecoration: "none",
    "& small": {
      color: "#777",
      fontSize: "65%",
      fontWeight: "400",
      lineHeight: "1",
    },
  },
  selectedImage: {
    display: "flex",
    justifyContent: "start",
    alignItems: "flex-start",
    width: "100%",
    height: "30px",
    backgroundColor: "#787878",
    textOverflow: "ellipsis",
    overflow: "hidden",
    "&:hover": {
      background: "#E8E8E8",
    },
  },
  imageNameContainer: {
    display: "flex",
    flexDirection: "column",
    gap: "10px",
    width: "100%",
    height: "100%",
  },
  imageName: {
    display: "flex",
    justifyContent: "start",
    alignItems: "flex-start",
    width: "95%",
    textOverflow: "ellipsis",
    overflow: "hidden",
    background: "#D0D0D0",
    borderRight: "solid",
    borderWidth: "1vw",
    borderRightColor: "#26b656",
    "&:hover": {
      background: "#E8E8E8",
    },
  },
  imageNameWrong: {
    display: "flex",
    justifyContent: "start",
    alignItems: "flex-start",
    width: "95%",
    textOverflow: "ellipsis",
    overflow: "hidden",
    background: "#D0D0D0",
    borderRight: "solid",
    borderWidth: "1vw",
    Right: "1vw",
    borderRightColor: "#581123",
    "&:hover": {
      background: "#581123",
    },
  },
  select: {
    "&:after": {
      borderColor: "#FEC260",
    },
  },
});

export default function ImageWrapper({
  isDragging,
  onImageUpload,
  dragProps,
  onImageRemoveAll,
  onImageUpdate,
  onImageRemove,
  mode,
  images,
  predictions,
  setPredictions,
  imageLabels,
  setImageLabels,
}) {
  const classes = useStyles();

  const [selectedImage, setSeletectImage] = useState("");
  const [selectedModel, setSelectedModel] = useState(0);

  const [checked, setChecked] = useState(false);

  useEffect(() => {
    if (predictions.get(selectedImage) !== undefined) {
      setChecked(predictions.get(selectedImage).correct);
    }
  }, [selectedImage, predictions]);

  const handleCheck = (event) => {
    predictions.get(selectedImage).correct = event.target.checked;
    setChecked(!checked);
    //send request for the server to use that model (we should load both for speed of prediction)
  };

  const handleAddToDataset = async () => {
    const myElement = document.getElementById("retrainbutton");
    myElement.style.color = "#000000";
    myElement.style.backgroundColor = "#C0A9BD";
    myElement.style.display = "Hallo";

    let alert = false;
    images.forEach((image) => {
      if (imageLabels.get(image.file.name) === undefined) {
        alert = true;
      }
    });
    if (alert === true) {
      window.alert("Make sure all images are labeled!");
    }
    images.forEach(async (image) => {
      if (imageLabels.get(image.file.name) === undefined) {
        alert = true;
      }

      const formData = new FormData();

      formData.append("file", image.file);
      // const file = formData.get("file");
      const options = {
        method: "POST",
        mode: "no-cors",
        //files: image.file,
        body: formData,
      };
      await fetch("/api/append_data", options)
        .then(
          (response) => {
            return response.json();
          } // if the response is a JSON object
        )
        .then(
          (success) => success // Handle the success response object
        )
        .catch(
          (error) => error // Handle the error response object
        );
      const options2 = {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          file_name: image.file.name,
          label: imageLabels.get(image.file.name),
        }),
      };
      await fetch("/api/append_label", options2)
        .then(
          (response) => {
            return response.json();
          } // if the response is a JSON object
        )
        .then(
          (success) => success // Handle the success response object
        )
        .catch(
          (error) => error // Handle the error response object
        );
    });
  };
  const handleModelSelection = async (event) => {
    setSelectedModel(event.target.value);
    const options = {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(event.target.value === 0 ? "transfer" : "selfsup"),
    };
    await fetch("/api/mode", options)
      .then(
        (response) => {
          return response.json();
        } // if the response is a JSON object
      )
      .then(
        (success) => success // Handle the success response object
      )
      .catch(
        (error) => error // Handle the error response object
      );
  };

  const handleExport = async () => {
    const link = document.createElement("a");

    // make request transformation =

    const response = await fetch("/api/export")
      .then((res) => {
        return res;
      })
      .then((data) => {
        return data.json();
      });
    let file = {
      annotations: JSON.parse(response),
    };
    let output = new TextEncoder().encode(JSON.stringify(file));

    link.download = `Labels_Data.json`;
    link.href = URL.createObjectURL(new Blob([output]));
    link.click();
  };
  const saveLabel = (imageName, event) => {
    setImageLabels((prevState) => {
      const newState = prevState.set(imageName, id2label(event.target.value));
      return newState;
    });
    const options = {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        file_name: imageName,
        label: id2label(event.target.value),
      }),
    };
    fetch("/api/label", options)
      .then(
        (response) => response.json() // if the response is a JSON object
      )
      .then(
        (success) => console.log(success) // Handle the success response object
      )
      .catch(
        (error) => console.log(error) // Handle the error response object
      );
  };

  const handleRetrain = async () => {
    const options = {
      method: "GET",
      mode: "no-cors",
    };
    let response = await fetch("/api/retrain", options)
      .then(
        (response) => {
          return response.json();
        } // if the response is a JSON object
      )
      .then(
        (success) => success // Handle the success response object
      )
      .catch(
        (error) => error // Handle the error response object
      );
    console.log(response);
  };
  const onPredict = async (image) => {
    const formData = new FormData();

    formData.append("file", image.file);
    // const file = formData.get("file");
    const options = {
      method: "POST",
      mode: "no-cors",
      body: formData,
    };
    let response = await fetch("/api/predict", options)
      .then(
        (response) => {
          return response.json();
        } // if the response is a JSON object
      )
      .then(
        (success) => success // Handle the success response object
      )
      .catch(
        (error) => error // Handle the error response object
      );
    setPredictions((prevState) => {
      const newState = prevState.set(response.file_name, {
        label: response.label,
        correct: true,
      });
      return newState;
    });
  };
  const id2label = (id) => {
    switch (id) {
      case 0:
        return "Scratch";
      case 1:
        return "Rim";
      case 2:
        return "Dent";
      case 3:
        return "Other";
      case 4:
        return "Not Defined";
      default:
        return "Not Defined";
    }
  };

  const label2id = (label) => {
    switch (label) {
      case "Scratch":
        return 0;
      case "Rim":
        return 1;
      case "Dent":
        return 2;
      case "Other":
        return 3;
      case "Not Defined":
        return 4;
      default:
        return "Not Defined";
    }
  };

  return (
    <div className="upload__image-wrapper alignnext">
      <Container class="paddingButton">
        <ButtonGroup
          style={{
            backgroundColor: "#C0A9BD",
            fontColor: "#F4F2F3",
          }}
          variant="filled"
        >
          <Button
            title="Upload your pictures here."
            style={{
              fontSize: "1vw",
              height: "2vw",
            }}
            onClick={onImageUpload}
            {...dragProps}
          >
            <UploadIcon
              style={{
                fontSize: "1.2vw",
              }}
            ></UploadIcon>
            Click or Drop here
          </Button>
          &nbsp;
          <Button
            title="Delete all pictures here."
            style={{
              fontSize: "1vw",
              height: "2vw",
            }}
            onClick={onImageRemoveAll}
          >
            <DeleteIcon
              style={{
                fontSize: "1.2vw",
              }}
            ></DeleteIcon>{" "}
            Remove all images
          </Button>
        </ButtonGroup>
      </Container>
      <br></br>
      <div class="textdiv6 padding ">
        {images.length !== 0 && (
          <GridContainer>
            <GridItem xs={12} sm={6} md={4}>
              <Card>
                {mode === "Prediction" && (
                  <CardHeader color="info">
                    <h4
                      className={classes.cardTitleWhite}
                      style={{
                        fontSize: "1vw",
                        fontFamily: "Titillium Web",
                        fontWeight: "lighter",
                      }}
                    >
                      A list of your pictures will be displayed here. Click on
                      one to display it on the right: <br></br>
                      Each picture has either a RED or GREEN tag on the right.
                      RED means that there is no label yet.
                      <br></br> <br></br>Press PREDICT to get a label.
                    </h4>
                  </CardHeader>
                )}
                {mode === "Labeling" && (
                  <CardHeader color="info">
                    <h4
                      className={classes.cardTitleWhite}
                      style={{
                        fontSize: "1vw",
                        fontFamily: "Titillium Web",
                        fontWeight: "lighter",
                      }}
                    >
                      A list of your pictures will be displayed here. Click on
                      one to display on the right: <br></br>
                      Each picture has either a RED or GREEN tag on the right.
                      RED means that there is no Label yet. <br></br> <br></br>
                      Select a label to turn it GREEN.
                    </h4>
                  </CardHeader>
                )}
                <CardBody>
                  <div className={classes.imageNameContainer}>
                    {images.map((image, index) => (
                      <>
                        <Button
                          className={
                            image.file.name === selectedImage
                              ? classes.selectedImage
                              : (predictions.get(image.file.name) !==
                                  undefined &&
                                  predictions.get(image.file.name).correct ===
                                    true) ||
                                imageLabels.get(image.file.name) !== undefined
                              ? classes.imageName
                              : classes.imageNameWrong
                          }
                          style={{
                            margin: "2px",
                            backgroundColor: "#C0A9BD",
                            height: "4vh",
                            padding: "0.5vh",
                            fontSize: "2vh",
                          }}
                          variant="raised"
                          onClick={() => {
                            setSeletectImage(image.file.name);
                          }}
                        >
                          {image.file.name}
                        </Button>
                      </>
                    ))}
                  </div>
                </CardBody>
              </Card>
            </GridItem>
            <GridItem xs={12} sm={12} md={8}>
              <Card>
                {mode === "Prediction" && (
                  <CardHeader color="info">
                    <h4
                      className={classes.cardTitleWhite}
                      style={{
                        fontSize: "1vw",
                        fontFamily: "Titillium Web",
                        fontWeight: "lighter",
                      }}
                    >
                      Select one of our two models by using the MODEL button.
                      Press PREDICT to get your label.<br></br>
                      If the label is correct, tick the box to add it to your
                      database.
                      <br></br> If it is wrong, switch to the LABELING tab to
                      correct the label.
                    </h4>
                  </CardHeader>
                )}
                {mode === "Labeling" && (
                  <CardHeader color="info">
                    <h4
                      className={classes.cardTitleWhite}
                      style={{
                        fontSize: "1vw",
                        fontFamily: "Titillium Web",
                        fontWeight: "lighter",
                      }}
                    >
                      Choose the correct label by pressing the drop-down button
                      LABEL. If you have uploaded more than one picture select
                      each one and align the appropriate label. Once you are
                      done, press EXPORT LABELS. <br></br> <br></br>For more
                      information about the buttons underneath, hover over them
                      for some seconds. <br></br>
                    </h4>
                  </CardHeader>
                )}

                <CardBody>
                  {images
                    .filter((image, index) => image.file.name === selectedImage)
                    .map((image, index) => (
                      <div
                        key={index}
                        className="row"
                        max-height="100px"
                        max-width="100px"
                      >
                        <div key={index} className="image-item">
                          <div className="image-item__btn-wrapper">
                            <GridContainer>
                              <GridItem xs={12} sm={12} md={12}>
                                {" "}
                                <ButtonGroup ariant="contained">
                                  <Button
                                    title="Delete the selected picture here."
                                    style={{
                                      backgroundColor: "#FFFFFF",
                                      fontSize: "1vw",
                                      height: "2vw",
                                    }}
                                    onClick={() => onImageRemove(index)}
                                  >
                                    <DeleteIcon
                                      style={{
                                        fontSize: "1.2vw",
                                      }}
                                    ></DeleteIcon>
                                    Delete
                                  </Button>
                                  {mode === "Prediction" && (
                                    <div>
                                      <div className={classes.ExportContainer}>
                                        <FormControl>
                                          <InputLabel id="input-label">
                                            Model
                                          </InputLabel>
                                          <Select
                                            style={{
                                              backgroundColor: "#FFFFFF",
                                              fontSize: "1vw",
                                              height: "2vw",
                                            }}
                                            className={classes.tabs}
                                            labelId="input-label"
                                            id="selected_label"
                                            defaultValue={1}
                                            value={selectedModel}
                                            displayEmpty
                                            label="Model"
                                            onChange={(event) => {
                                              handleModelSelection(event);
                                            }}
                                          >
                                            <MenuItem value={0}>
                                              {"EfficientNetV2ILSVRC"}
                                            </MenuItem>
                                            <MenuItem value={1}>
                                              {"SimCLR"}
                                            </MenuItem>
                                          </Select>
                                        </FormControl>
                                      </div>
                                    </div>
                                  )}
                                  {mode === "Prediction" && (
                                    <Button
                                      title="Start the magic now!"
                                      style={{
                                        backgroundColor: "#FFFFFF",
                                        fontSize: "1vw",
                                        height: "2vw",
                                      }}
                                      onClick={() => onPredict(image)}
                                    >
                                      Predict
                                    </Button>
                                  )}
                                  {mode === "Labeling" && (
                                    <div>
                                      <div className={classes.ExportContainer}>
                                        <>
                                          <FormControl>
                                            <InputLabel id="input-label">
                                              Label
                                            </InputLabel>
                                            <Select
                                              style={{
                                                backgroundColor: "#FFFFFF",
                                                fontSize: "1vw",
                                                height: "2vw",
                                              }}
                                              className={classes.tabs}
                                              labelId="input-label"
                                              id="selected_label"
                                              defaultValue={4}
                                              value={
                                                imageLabels.get(
                                                  image.file.name
                                                ) !== undefined
                                                  ? label2id(
                                                      imageLabels.get(
                                                        image.file.name
                                                      )
                                                    )
                                                  : 4
                                              }
                                              displayEmpty
                                              label="Label"
                                              onChange={(event) => {
                                                saveLabel(
                                                  image.file.name,
                                                  event
                                                );
                                              }}
                                            >
                                              <MenuItem value={0}>
                                                {"Scratch"}
                                              </MenuItem>
                                              <MenuItem value={1}>
                                                {"Rim"}
                                              </MenuItem>
                                              <MenuItem value={2}>
                                                {"Dent"}
                                              </MenuItem>
                                              <MenuItem value={3}>
                                                {"Other"}
                                              </MenuItem>
                                              <MenuItem value={4}>
                                                {"Not Defined"}
                                              </MenuItem>
                                            </Select>
                                          </FormControl>
                                        </>
                                      </div>
                                    </div>
                                  )}{" "}
                                  {mode === "Labeling" && (
                                    <Button
                                      title="Add your selected image to our dataset, afterwards click retrain to check out the upgraded model!"
                                      style={{
                                        fontSize: "1vw",
                                        height: "2vw",
                                      }}
                                      ariant="contained"
                                      onClick={handleAddToDataset}
                                    >
                                      Add to Dataset
                                    </Button>
                                  )}
                                  {mode === "Labeling" && (
                                    <Button
                                      id="retrainbutton"
                                      title="Retrain our model. Move over to predict to use the upgraded model!"
                                      style={{
                                        fontSize: "1vw",
                                        height: "2vw",
                                        color: "#ffffff",
                                        backroundColor: "#ffffff",
                                      }}
                                      ariant="contained"
                                      onClick={handleRetrain}
                                    >
                                      Retrain Model
                                    </Button>
                                  )}
                                  {mode === "Labeling" && (
                                    <Button
                                      title="A json file
                                            called labels_data will be generated with the image labels
                                             and titles that will be downloaded automatically."
                                      style={{
                                        backgroundColor: "#C0A9BD",
                                        fontSize: "1vw",
                                        height: "2vw",
                                      }}
                                      ariant="contained"
                                      onClick={handleExport}
                                    >
                                      Export Labels
                                    </Button>
                                  )}
                                  <Slide
                                    direction="up"
                                    in={
                                      predictions.get(selectedImage) !==
                                        undefined &&
                                      selectedImage === image.file.name &&
                                      mode === "Prediction"
                                    }
                                    mountOnEnter
                                    unmountOnExit
                                  >
                                    <div className={"textPara"}>
                                      <TextField
                                        type="text"
                                        value={
                                          predictions.get(selectedImage) !==
                                          undefined
                                            ? predictions.get(selectedImage)
                                                .label
                                            : ""
                                        }
                                        variant="outlined"
                                        inputProps={{
                                          readOnly: true,
                                          style: {
                                            textAlign: "center",
                                            height: "0vw",
                                            fontSize: "1.2vw",
                                            width: "6vw",
                                            paddingTop: "1vw",
                                            paddingBottom: "1vw",
                                            backgroundColor: "#C0A9BD",
                                          },
                                        }}
                                      />
                                      <div class="stylecorrect">
                                        <FormControlLabel
                                          control={
                                            <Checkbox
                                              onClick={handleCheck}
                                              color={"#2A0944"}
                                              checked={
                                                predictions.get(
                                                  selectedImage
                                                ) !== undefined
                                                  ? predictions.get(
                                                      selectedImage
                                                    ).correct
                                                  : false
                                              }
                                            />
                                          }
                                          label={
                                            <Box
                                              component="div"
                                              alignContent={"top"}
                                              fontSize={"1vw"}
                                            >
                                              CORRECT
                                            </Box>
                                          }
                                        />
                                      </div>
                                    </div>
                                  </Slide>
                                </ButtonGroup>
                              </GridItem>{" "}
                            </GridContainer>
                          </div>
                          <div class="gallery ">
                            <div class="iframe-behaelter123">
                              <img
                                className={classes.imageWrapper}
                                src={image.data_url}
                                alt=""
                              />
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                </CardBody>
              </Card>
            </GridItem>
            <GridItem>
              {mode === "Prediction" && (
                <Button onClick={() => onPredict()}></Button>
              )}
            </GridItem>
          </GridContainer>
        )}
        <GridItem xs={12} sm={12} md={12}>
          <p>
            <br></br> <br></br> <br></br> <br></br> <br></br> <br></br>
            <br></br> <br></br> <br></br> <br></br> <br></br> <br></br>
            <br></br> <br></br> <br></br> <br></br> <br></br> <br></br>{" "}
            <br></br> <br></br>
            <br></br>
            <br></br> <br></br>
            <br></br>
            <br></br>
            <br></br>
            <br></br>
            <br></br>
            <br></br>
          </p>
        </GridItem>
      </div>
    </div>
  );
}
