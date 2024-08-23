import React, {useState,useEffect} from "react";
import {
    Box, Button, Flex, Input, InputGroup,
    Modal, ModalBody, ModalCloseButton, 
    ModalContent, ModalHeader, ModalFooter,
    ModalOverlay, Stack, Text, useDisclosure
} from "@chakra-ui/react"

const OutputsContext = React.createContext({
    outputs: [], fetchOutput: () => {}
})

// fetching algorithm
export default function Todos(){
    const[outputs, setOutputs] = useState([])
    const fetchOutputs = async() => {
        const response = await fetch("http://localhost:8000/get_predictions")
        const outputs = await response.json()
        setOutputs(outputs.data)
    }
useEffect(()=> {
    fetchOutputs()
}, [])

return(
    <OutputsContext.Provider value = {{outputs, fetchOutputs}}>
        <Stack spacing={5}>
            {outputs.map((output)=>(
                <b>{output.item}</b>
            ))}
        </Stack>
    </OutputsContext.Provider>
)

}