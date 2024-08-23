import React, { createRef } from "react";
import { createRoot } from 'react-dom/client';
import { ChakraProvider } from "@chakra-ui/react";

import Header from "./components/Header";
// import Todos

function App(){
  return(
    <ChakraProvider>
      <Header/>
      <Output/>
    </ChakraProvider>
  )
}

const rootelement = document.getElementById("root")
const root = createRoot(rootelement)
root.render(<App tab="home"/>)