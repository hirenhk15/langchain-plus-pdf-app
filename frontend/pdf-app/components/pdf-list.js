import styles from '../styles/pdf-list.module.css';
import { useState, useEffect, useCallback, useRef } from 'react';
import { debounce } from 'lodash';
import PDFComponent from './pdf';

export default function PDFList() {
    const [pdfs, setPdfs] = useState([]);
    const [selectedFile, setSelectedFile] = useState(null);
    const [filter, setFilter] = useState();
    const didFetchRef = useRef(false);
    const [question, setQuestion] = useState('');
    const [answer, setAnswer] = useState(null); // Define qa state
    const [isAnswerVisible, setIsAnswerVisible] = useState(false); //Track the visibility of qa box

    useEffect(() => {
        if (!didFetchRef.current) {
            didFetchRef.current = true;
            fetchPdfs();
        }
    }, []);

    async function fetchPdfs(selected) {
        let path = '/pdfs';
        if (selected !== undefined) {
            path = `/pdfs?selected=${selected}`;
        }
        const res = await fetch(process.env.NEXT_PUBLIC_API_URL + path)
        const json = await res.json();
        setPdfs(json);
    }

    const debouncedUpdatePdf = useCallback(debounce((pdf, fieldChanged) => {
        updatePdf(pdf, fieldChanged);
    }, 500), []);

    function handlePdfChange(e, id) {
        const target = e.target;
        const value = target.type === 'checkbox' ? target.checked : target.value;
        const name = target.name;
        const copy = [...pdfs];
        const idx = pdfs.findIndex((pdf) => pdf.id === id);
        const changedPdf = { ...pdfs[idx], [name]: value };
        copy[idx] = changedPdf;
        debouncedUpdatePdf(changedPdf, name);
        setPdfs(copy);
    }

    async function updatePdf(pdf, fieldChanged) {
        const body_data = JSON.stringify(pdf);
        const url = process.env.NEXT_PUBLIC_API_URL + `/pdfs/${pdf.id}`

        await fetch(url, {
            method: 'PUT',
            body: body_data,
            headers: { 'Content-Type': 'application/json' }
        });
    }

    async function handleDeletePdf(id) {
        const res = await fetch(process.env.NEXT_PUBLIC_API_URL + `/pdfs/${id}`, {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' }
        });

        if (res.ok) {
            const copy = pdfs.filter((pdf) => pdf.id !== id);
            setPdfs(copy);
        }
    }

    const handleFileChange = (event) => {
        setSelectedFile(event.target.files[0]);
    };

    const handleUpload = async (event) => {
        event.preventDefault();
        if (!selectedFile) {
            alert("Please select file to load.");
            return;
        }

        const formData = new FormData();
        formData.append("file", selectedFile);

        const response = await fetch(process.env.NEXT_PUBLIC_API_URL + "/pdfs/upload", {
            method: "POST",
            body: formData
        });

        if (response.ok) {
            const newPdf = await response.json();
            setPdfs([...pdfs, newPdf]);
        } else {
            alert("Error loading file.");
        }
    };

    function handleFilterChange(value) {
        setFilter(value);
        fetchPdfs(value);
    }

    // QA functions
    // The following function is added to get answer given question using LangChain
    async function getAnswer(question) {
        const copy = [...pdfs];
        const sidx = pdfs.map((pdf, index) => pdf.selected === true ? index : -1).filter(index => index !== -1);
        
        // Ensure that pdfs is defined and has the correct length
        if (sidx.length === 1) {
            try {
                const selectedPdf = { ...pdfs[sidx] };
                const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/pdfs/qa-pdf/${selectedPdf.id}`, {
                    method: 'POST',
                    body: JSON.stringify({'question': question}),
                    headers: {
                        'Content-Type': 'application/json'
                    },
                });
                
                if (res.ok) {
                    const data = await res.json();
                    setAnswer(data.answer);
                    setIsAnswerVisible(true); // Show the answer box when answer is generated
                } else {
                    alert("Failed to fetch the answer. Please try again.");
                }
            } catch (error) {
                console.error("Error fetching the answer:", error);
                alert("An error occurred. Please try again.");
            }
        } else {
            alert("Select only one file to get the answer.");
        }
    }

    // Function to close the answer box
    function closeAnswerBox() {
        setIsAnswerVisible(false);
    }

    return (
        <div className={styles.container}>
            <div className={styles.mainInputContainer}>
                <form onSubmit={handleUpload}>
                    <input className={styles.mainInput} type="file" accept=".pdf" onChange={handleFileChange} />
                    <button className={styles.loadBtn} type="submit">Load PDF</button>
                </form>
            </div>
            {!pdfs.length && <div>Loading...</div>}
            {pdfs.map((pdf) => (
                <PDFComponent key={pdf.id} pdf={pdf} onDelete={handleDeletePdf} onChange={handlePdfChange} />
            ))}
            <div className={styles.filters}>
                <button className={`${styles.filterBtn} ${filter === undefined && styles.filterActive}`} onClick={() => handleFilterChange()}>See All</button>
                <button className={`${styles.filterBtn} ${filter === true && styles.filterActive}`} onClick={() => handleFilterChange(true)}>See Selected</button>
                <button className={`${styles.filterBtn} ${filter === false && styles.filterActive}`} onClick={() => handleFilterChange(false)}>See Not Selected</button>
            </div>
            <div className={styles.qacontainer}>
                <p className={styles.qatitle}>Enter your question</p>
                <input className={styles.question} autoComplete='off' name="name" type="text" value={question} onChange={(e) => { setQuestion(e.currentTarget.value); }}></input>
                <button className={styles.loadBtn} onClick={() => getAnswer(question)}>Submit</button>
                {isAnswerVisible && (
                    <div className={styles.answer}>
                        <p>{answer}</p>
                    </div>
                    // <div className={styles.answerBox}>
                    //     <button className={styles.closeButton} onClick={closeAnswerBox}>
                    //         &times; {/* Add a close icon */}
                    //     </button>
                    //     <div className={styles.answer}>
                    //         <p>{answer}</p>
                    //     </div>
                    // </div>
                )}
            </div>
        </div>
    );
}