// https://mui.com/material-ui/react-tabs/
import React, { useState } from "react";
import { Map } from "immutable";
import "./App.css";
import UploadFile from "./components/UploadFile";
import MainPage from "./components/MainPage";
import ModellingPage from "./components/ModellingPage";
import TeamPage from "./components/TeamPage";
import Typography from "@material-ui/core/Typography";
import PropTypes from "prop-types";
import Tabs from "@mui/material/Tabs";
import Tab from "@mui/material/Tab";
import Deepold from "./images_website/Deepold_header.png";
import AlphaDamage from "./images_website/white_AD.png";

import { makeStyles } from "@material-ui/core/styles";

const useStyles = makeStyles({
  tabs: {
    "& .MuiTabs-indicator": {
      backgroundColor: "#C0A9BD",
      height: "0.3vw",
      position: "fixed",
      top: "3.5vw",
      marginLeft: "35vw",
    },
    "& .MuiTab-root.Mui-selected": {
      color: "#FFFFFF",
    },
  },
});

function TabPanel(props) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`vertical-tabpanel-${index}`}
      aria-labelledby={`vertical-tab-${index}`}
      {...other}
    >
      {value === index && <Typography>{children}</Typography>}
    </div>
  );
}
TabPanel.propTypes = {
  children: PropTypes.node,
  index: PropTypes.number.isRequired,
  value: PropTypes.number.isRequired,
};

function a11yProps(index) {
  return {
    id: `vertical-tab-${index}`,
    "aria-controls": `vertical-tabpanel-${index}`,
  };
}

function App() {
  // const [mode] = useState("Main");
  const [value, setValue] = React.useState(0);
  const [images, setImages] = useState([]);
  const [predictions, setPredictions] = useState(new Map());
  const [imageLabels, setImageLabels] = useState(new Map());

  const handleChange = (event, newValue) => {
    setValue(newValue);
    window.scroll({
      top: 0,
      left: 0,
      behavior: "smooth",
    });
  };
  const classes = useStyles();

  return (
    <>
      <body>
        <main>
          <meta
            name="viewport"
            content="width=device-width, initial-scale=1.0"
          ></meta>

          <div class="headerRight">
            <p>
              <img width="100%" object-fit="contain" src={Deepold} alt="logo" />
            </p>
          </div>
          <div class="headerLeft">
            <p>
              <img
                width="100%"
                object-fit="contain"
                src={AlphaDamage}
                alt="logo2"
              />
            </p>
          </div>

          <div className="navbar">
            <div className="content">
              <Tabs
                className={classes.tabs}
                orientation="horizontal"
                variant="scrollable"
                value={value}
                scrollButtons="auto"
                onChange={handleChange}
                style={{
                  display: "flex",
                  color: "#F4F2F3",
                }}
              >
                <Tab
                  style={{
                    display: "flex",
                    justifyContent: "flex-start",
                    alignItems: "center",
                    color: "#F4F2F3",
                    fontSize: "1.4vw",
                    height: "1vw",
                    fontFamily: "Titillium Web",
                    fontWeight: "lighter",
                  }}
                  label="Home"
                  {...a11yProps(0)}
                ></Tab>
                <Tab
                  style={{
                    display: "flex",
                    justifyContent: "flex-start",
                    alignItems: "center",
                    color: "#F4F2F3",
                    fontSize: "1.4vw",
                    height: "1vw",
                    fontFamily: "Titillium Web",
                    fontWeight: "lighter",
                    marginLeft: "0.5vw",
                  }}
                  label="Prediction"
                  {...a11yProps(1)}
                />
                <Tab
                  style={{
                    display: "flex",
                    justifyContent: "flex-start",
                    alignItems: "center",
                    color: "#F4F2F3",
                    fontSize: "1.4vw",
                    height: "1vw",
                    fontFamily: "Titillium Web",
                    fontWeight: "lighter",
                    marginLeft: "0.5vw",
                  }}
                  label="Labeling"
                  {...a11yProps(2)}
                />
                <Tab
                  style={{
                    display: "flex",
                    justifyContent: "flex-start",
                    alignItems: "center",
                    color: "#F4F2F3",
                    fontSize: "1.4vw",
                    height: "1vw",
                    fontFamily: "Titillium Web",
                    fontWeight: "lighter",
                    marginLeft: "0.5vw",
                  }}
                  label="Research"
                  {...a11yProps(3)}
                />
                <Tab
                  style={{
                    display: "flex",
                    justifyContent: "flex-start",
                    alignItems: "center",
                    color: "#F4F2F3",
                    fontSize: "1.4vw",
                    height: "1vw",
                    fontFamily: "Titillium Web",
                    fontWeight: "lighter",
                    marginLeft: "0.5vw",
                  }}
                  label="Team"
                  {...a11yProps(4)}
                />
              </Tabs>
            </div>
          </div>
          <div class="contentRight">
            <TabPanel value={value} index={0}>
              <MainPage></MainPage>
            </TabPanel>

            <TabPanel value={value} index={1}>
              <UploadFile
                maxNumber={20}
                mode={"Prediction"}
                images={images}
                setImages={setImages}
                predictions={predictions}
                setPredictions={setPredictions}
                imageLabels={imageLabels}
                setImageLabels={setImageLabels}
              ></UploadFile>
            </TabPanel>

            <TabPanel value={value} index={2}>
              <UploadFile
                maxNumber={20}
                mode={"Labeling"}
                images={images}
                setImages={setImages}
                predictions={predictions}
                setPredictions={setPredictions}
                imageLabels={imageLabels}
                setImageLabels={setImageLabels}
              ></UploadFile>
            </TabPanel>
            <TabPanel value={value} index={3}>
              <ModellingPage></ModellingPage>
            </TabPanel>
            <TabPanel value={value} index={4}>
              <TeamPage></TeamPage>
            </TabPanel>
          </div>

          <div class="footerCop ">
            <div class="footerTextBig">
              {" "}
              Intelligent Solutions to Advance
              <br></br>Science and Benefit Humanity
            </div>

            <div class=" footerTextSmall ">
              {" "}
              <br></br>
              <br></br> <br></br>
              Deep.old © 2022 All Rights Reserved | Privacy Policy | Terms and
              Conditions | Alphadamage@deep-old.xyz
            </div>
          </div>
        </main>
      </body>
    </>
  );
}

export default App;
