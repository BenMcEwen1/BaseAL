import { useState, useEffect } from 'react';
import { getMedia } from '../utils/apiClient';

export default function Viewer({mediaID, setID}) {
    const [mediaData, setMediaData] = useState(null)

    useEffect(() => {
        if (mediaID !== null && mediaID !== undefined) {
            retrieveMedia()
        } else {
            setMediaData(null)
        }
    }, [mediaID])

    const retrieveMedia = async () => {
        try {
            const data = await getMedia(mediaID);
            // data contains { audio: "data:audio/...", spectrogram: "data:image/png;..." }
            setMediaData(data);
        } catch (err) {
            console.error('Failed to load data:', err);
        }
    }

    return (
        <div style={{
            position: 'absolute',
            margin: '50px',
            bottom: '0px',
            right: '0px',
            zIndex: '10'
        }}>
            {mediaData &&
                <div style={{
                    backgroundColor: "#303039ff",
                    borderRadius: '15px',
                    padding: '15px',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '10px',
                    minWidth: '400px'
                }}>
                    {/* Close button */}
                    <div style={{display: 'flex', justifyContent: 'flex-end'}}>
                        <button
                            style={{
                                width: '30px',
                                height: '30px',
                                backgroundColor: 'transparent',
                                borderRadius: '50%',
                                border: 'none',
                                color: '#fff',
                                cursor: 'pointer',
                                fontSize: '14px'
                            }}
                            onClick={() => setID(null)}
                        >
                            ✖
                        </button>
                    </div>

                    {/* Spectrogram */}
                    {mediaData.spectrogram && (
                        <div style={{
                            backgroundColor: '#2a2a2a',
                            borderRadius: '8px',
                            overflow: 'hidden'
                        }}>
                            <img
                                src={mediaData.spectrogram}
                                alt="Audio Spectrogram"
                                style={{width: '100%', display: 'block'}}
                            />
                        </div>
                    )}

                    {/* Audio Player */}
                    {mediaData.audio && (
                        <div style={{
                            // backgroundColor: "#2a2a2a",
                            borderRadius: '8px',
                            padding: '10px'
                        }}>
                            <audio
                                controls
                                src={mediaData.audio}
                                style={{width: '100%'}}
                                autoPlay
                            />
                        </div>
                    )}
                </div>
            }
        </div>
    )
}