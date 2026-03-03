import { useState, useEffect } from 'react';

const teamMembers = [
  {
    id: 1,
    name: 'Ben McEwen',
    affiliation: 'Postdoctoral researcher, Tilburg University, Netherlands',
    description: 'Ben is a Postdoctoral Researcher in AI and Biodiversity applying Active Learning to biodiversity monitoring at a transnational scale. Previously, Ben researched Active Learning methods for at-risk and invasive species detection',
    contact: 'benmcewen@outlook.com',
    googleScholar: 'https://scholar.google.com/citations?hl=en&user=x47JZUkAAAAJ&view_op=list_works&sortby=pubdate',
    website: 'https://www.benmcewen-phd.com/',
    img: '/profiles/ben_mcewen.jpg'
  },
  {
    id: 2,
    name: 'Lukas Rauch',
    affiliation: 'PhD Candidate, Kassel University, Germany',
    description: '',
    googleScholar: 'https://scholar.google.com/citations?hl=en&user=bB2A6e0AAAAJ&view_op=list_works&sortby=pubdate',
    // website: 'https://example.com/',
    img: '/profiles/lukas_rauch.jpg'
  },
  {
    id: 3,
    name: 'Marek Herde',
    affiliation: 'PhD Candidate, Kassel University, Germany',
    description: '',
    googleScholar: 'https://scholar.google.com/citations?hl=en&user=pwRDfMQAAAAJ&view_op=list_works&sortby=pubdate',
    // website: 'https://example.com/',
    img: '/profiles/marek_herde.JPG'
  },
  {
    id: 4,
    name: 'Shiqi Zhang',
    affiliation: 'PhD Candidate, Tampere University, Finland',
    description: "Shiqi Zhang is a PhD candidate in the Audio Research Group at Tampere University. His research focuses on developing Active Learning methods to minimize the manual annotation effort required for bioacoustic data analysis. He is a member of the Bioacoustic AI project, funded by the European Union's Marie Skłodowska-Curie Action.",
    googleScholar: 'https://scholar.google.com/citations?hl=en&user=fnOCg-8AAAAJ&view_op=list_works&sortby=pubdate',
    // website: 'https://example.com/',
    img: '/profiles/shiqi_zhang.jpg'
  },
  {
    id: 5,
    name: 'Rupa Kurinchi-Vendhan',
    affiliation: 'PhD Candidate, Massachusetts Institute of Technology',
    description: 'Rupa Kurinchi-Vendhan is a second-year Ph.D. student in the MIT Department of Electrical Engineering and Computer Science, advised by Professor Sara Beery and supported by the NSF Graduate Research Fellowship and the MIT Tina Chan Fellowship. Her research focuses on expert-in-the-loop systems for scientific discovery in domains with sparse and low-quality data, across modalities.',
    googleScholar: 'https://scholar.google.com/citations?hl=en&user=YY9cf7sAAAAJ&view_op=list_works&sortby=pubdate',
    website: 'https://rupakv.com/',
    img: '/profiles/rupa_kurinchi_vendhan.jfif'
  },
  {
    id: 6,
    name: 'John Martinsson',
    affiliation: 'RISE Research Institute of Sweden',
    description: '',
    googleScholar: 'https://scholar.google.com/citations?hl=en&user=sAMIwlMAAAAJ&view_op=list_works&sortby=pubdate',
    website: 'https://johnmartinsson.org/',
    img: '/profiles/john_martinsson.jpg'
  },
  {
    id: 7,
    name: 'Sara Beery',
    affiliation: 'Associate Professor, Massachusetts Institute of Technology',
    description: 'Dr. Sara Beery is the Homer A. Burnell Assistant Professor in the MIT Department of Electrical Engineering and Computer Science. Her research focuses on building computer vision methods that enable global-scale environmental and biodiversity monitoring across data modalities, and her work has been recognized with a Schmidt Sciences AI2050 Early Career Fellowship, an NSF CAREER Grant, and the Amori Doctoral Prize.  She also works to increase access to AI skills through interdisciplinary capacity building and education, and has founded the AI for Conservation slack community, founded and directs the Workshop on Computer Vision Methods for Ecology, and co-leads the NSF/NSERC Global Center on AI and Biodiversity Change.',
    googleScholar: 'https://scholar.google.com/citations?user=Hbr4c10AAAAJ&hl=en&oi=ao',
    website: 'https://beerys.github.io/',
    img: '/profiles/sara_beery.jpeg'
  },
];

function ScholarIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 24a7 7 0 1 1 0-14 7 7 0 0 1 0 14zm0-24L0 9.5l4.838 3.94A8 8 0 0 1 12 9a8 8 0 0 1 7.162 4.44L24 9.5z"/>
    </svg>
  );
}

function WebsiteIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10"/>
      <line x1="2" y1="12" x2="22" y2="12"/>
      <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>
    </svg>
  );
}

function TeamMemberCard({ member }) {
  const [expanded, setExpanded] = useState(false);
  const [hovered, setHovered] = useState(false);

  return (
    <div
      style={{
        backgroundColor: 'rgba(255, 255, 255, 0.1)',
        borderRadius: '12px',
        padding: '40px',
        textAlign: 'center',
        cursor: 'pointer',
        transition: 'all 0.3s ease',
        flex: '1 1 calc(50% - 10px)',
        maxWidth: 'calc(50% - 10px)',
        minWidth: '250px',
        boxSizing: 'border-box',
        border: hovered ? '2px solid rgba(255, 255, 255, 0.5)' : '2px solid transparent',
        transform: hovered ? 'scale(1.02)' : 'scale(1)',
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      onClick={() => setExpanded(!expanded)}
    >
      {/* Profile Photo Circle */}
      <div style={{
        width: '100px',
        height: '100px',
        borderRadius: '50%',
        backgroundColor: 'rgba(255, 255, 255, 0.2)',
        margin: '0 auto 15px auto',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: '26px',
        color: 'rgba(255, 255, 255, 0.5)',
        overflow: 'hidden'
      }}>
        <img style={{width: '100px',height: '100px'}} src={member.img}></img>
      </div>

      {/* Name */}
      <h4 style={{
        margin: '0 0 5px 0',
        fontSize: '0.875rem',
        color: 'white'
      }}>
        {member.name}
      </h4>

      {/* Affiliation */}
      <p style={{
        margin: '0 0 10px 0',
        fontSize: '0.7rem',
        color: 'rgba(255, 255, 255, 0.7)',
        fontStyle: 'italic'
      }}>
        {member.affiliation}
      </p>

      {/* Link Icons */}
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        gap: '12px',
        marginBottom: '10px',
      }}>
        {member.googleScholar && (
          <a
            href={member.googleScholar}
            target="_blank"
            rel="noopener noreferrer"
            onClick={(e) => e.stopPropagation()}
            style={{
              color: 'rgba(255, 255, 255, 0.7)',
              transition: 'color 0.2s ease',
            }}
            title="Google Scholar"
          >
            <ScholarIcon />
          </a>
        )}
        {member.website && (
          <a
            href={member.website}
            target="_blank"
            rel="noopener noreferrer"
            onClick={(e) => e.stopPropagation()}
            style={{
              color: 'rgba(255, 255, 255, 0.7)',
              transition: 'color 0.2s ease',
            }}
            title="Personal Website"
          >
            <WebsiteIcon />
          </a>
        )}
      </div>

      {/* Contact */}
      {member.contact && (
        <p style={{
          margin: '0 0 10px 0',
          fontSize: '0.65rem',
          color: 'rgba(255, 255, 255, 0.6)',
        }}>
          {member.contact}
        </p>
      )}

      {/* Expandable Description */}
      <div style={{
        maxHeight: expanded ? '400px' : '0',
        overflow: 'hidden',
        transition: 'max-height 0.3s ease',
      }}>
        <p style={{
          margin: '10px 0 0 20px',
          fontSize: '0.7rem',
          color: 'rgba(255, 255, 255, 0.8)',
          textAlign: 'left',
          lineHeight: '1.5'
        }}>
          {member.description}
        </p>
      </div>

      {/* Expand indicator */}
      <span style={{
        fontSize: '0.6rem',
        color: 'rgba(255, 255, 255, 0.5)',
        marginTop: '8px',
        display: 'block'
      }}>
        {/* {expanded ? 'V' : null} */}
      </span>
    </div>
  );
}

function Timeline() {
  const launchDate = new Date('2026-04-01');
  const deadlineDate = new Date('2026-06-15');
  const [now, setNow] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setNow(new Date()), 60000);
    return () => clearInterval(timer);
  }, []);

  const getDaysUntil = (targetDate) => {
    const diff = targetDate - now;
    return Math.ceil(diff / (1000 * 60 * 60 * 24));
  };

  const daysToLaunch = getDaysUntil(launchDate);
  const daysToDeadline = getDaysUntil(deadlineDate);
  const hasLaunched = daysToLaunch <= 0;
  const hasClosed = daysToDeadline <= 0;

  const getBoxStyle = () => {
    if (hasClosed) {
      return {
        background: 'linear-gradient(135deg, rgba(226, 74, 74, 0.15), rgba(226, 74, 74, 0.05))',
        border: '1px solid rgba(226, 74, 74, 0.3)',
      };
    }
    if (hasLaunched) {
      return {
        background: 'linear-gradient(135deg, rgba(74, 180, 226, 0.15), rgba(74, 180, 226, 0.05))',
        border: '1px solid rgba(74, 180, 226, 0.3)',
      };
    }
    return {
      background: 'linear-gradient(135deg, rgba(74, 226, 144, 0.15), rgba(74, 226, 144, 0.05))',
      border: '1px solid rgba(74, 226, 144, 0.3)',
    };
  };

  const getLineGradient = () => {
    if (hasClosed) {
      return 'linear-gradient(90deg, rgba(74, 226, 144, 0.8), rgba(226, 74, 74, 0.8))';
    }
    return 'linear-gradient(90deg, rgba(74, 226, 144, 0.8), rgba(74, 180, 226, 0.8))';
  };

  return (
    <>
      <h2>Timeline</h2>

      <div style={{
        ...getBoxStyle(),
        borderRadius: '12px',
        padding: 'clamp(15px, 4vw, 30px) clamp(20px, 5vw, 40px)',
        marginBottom: '40px',
      }}>
        {/* Timeline row with circles */}
        <div style={{
          display: 'flex',
          alignItems: 'flex-start',
          justifyContent: 'space-between',
          position: 'relative',
        }}>
          {/* Connecting line - positioned at half the circle height/width */}
          <div style={{
            position: 'absolute',
            top: 'clamp(25px, 6vw, 40px)',
            left: 'clamp(25px, 6vw, 40px)',
            right: 'clamp(25px, 6vw, 40px)',
            height: '3px',
            background: getLineGradient(),
            zIndex: 0,
          }} />

          {/* Launch Date */}
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            zIndex: 1,
          }}>
            <div style={{
              width: 'clamp(50px, 12vw, 80px)',
              height: 'clamp(50px, 12vw, 80px)',
              borderRadius: '50%',
              background: '#1a2332',
              border: '3px solid rgba(74, 226, 144, 0.8)',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: hasLaunched ? '0 0 15px rgba(74, 226, 144, 0.4)' : 'none',
            }}>
              <span style={{ fontSize: 'clamp(0.9rem, 2vw, 1.2rem)', fontWeight: 'bold', color: '#4ae290' }}>1</span>
              <span style={{ fontSize: 'clamp(0.45rem, 1vw, 0.55rem)', color: 'rgba(255,255,255,0.7)', textTransform: 'uppercase' }}>Apr</span>
            </div>
            <span style={{
              fontSize: 'clamp(0.55rem, 1.2vw, 0.65rem)',
              fontWeight: '600',
              color: '#4ae290',
              textTransform: 'uppercase',
              letterSpacing: '1px',
              marginTop: '10px',
            }}>Launch</span>
          </div>

          {/* Close Date */}
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            zIndex: 1,
          }}>
            <div style={{
              width: 'clamp(50px, 12vw, 80px)',
              height: 'clamp(50px, 12vw, 80px)',
              borderRadius: '50%',
              background: '#1a2332',
              border: `3px solid ${hasClosed ? 'rgba(226, 74, 74, 0.8)' : 'rgba(74, 180, 226, 0.8)'}`,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: hasClosed ? '0 0 15px rgba(226, 74, 74, 0.4)' : (hasLaunched ? '0 0 15px rgba(74, 180, 226, 0.3)' : 'none'),
            }}>
              <span style={{ fontSize: 'clamp(0.9rem, 2vw, 1.2rem)', fontWeight: 'bold', color: hasClosed ? '#e24a4a' : '#4ab4e2' }}>15</span>
              <span style={{ fontSize: 'clamp(0.45rem, 1vw, 0.55rem)', color: 'rgba(255,255,255,0.7)', textTransform: 'uppercase' }}>Jun</span>
            </div>
            <span style={{
              fontSize: 'clamp(0.55rem, 1.2vw, 0.65rem)',
              fontWeight: '600',
              color: hasClosed ? '#e24a4a' : '#4ab4e2',
              textTransform: 'uppercase',
              letterSpacing: '1px',
              marginTop: '10px',
            }}>Deadline</span>
          </div>
        </div>

        {/* Countdown / Status below */}
        <div style={{
          textAlign: 'center',
          // marginTop: 'clamp(-35px, -6vw, -50px)',
          paddingTop: '15px',
        }}>
          {hasClosed ? (
            <span style={{
              fontSize: '0.8rem',
              fontWeight: 'bold',
              color: '#e24a4a',
            }}>
              Challenge Closed
            </span>
          ) : (
            <>
              <span style={{ color: 'rgba(255,255,255,0.6)', fontSize: '0.7rem' }}>
                {hasLaunched ? 'Submissions close in ' : 'Challenge launches in '}
              </span>
              <span style={{
                fontSize: '0.875rem',
                fontWeight: 'bold',
                color: hasLaunched ? '#4ab4e2' : '#4ae290',
              }}>
                {hasLaunched ? daysToDeadline : daysToLaunch} days
              </span>
            </>
          )}
        </div>
      </div>
    </>
  );
}

export default function BioDCASE() {
  return (
    <div style={{
      padding: '60px',
      maxWidth: '900px',
      margin: '0 auto',
      color: 'white'
    }}>
        {/* <h2 style={{fontSize: '14px', color: 'grey'}}>BioDCASE 2026</h2> */}

        <img src='/pages/AL4Bioacoustics.jpg' alt="" style={{width:"100%"}}></img>

        {/* <h1>Active Learning for Bioacoustics</h1> */}

        {/* <tag>Active Learning, Bioacoustics</tag> */}
        
        <h3>Description</h3>
        <p>A fundamental challenge across bioacoustics domains (terrestrial and marine) is the annotation of unlabelled data. Passive acoustic monitoring systems generate vast amounts of data, but only a small portion can be feasibly annotated by expert human annotators. Since model performance depends heavily on the quality and quantity of labelled data, this raises the following research question:</p> 
        <p style={{fontSize: '18px', 
                   color: "rgb(207, 207, 207)", 
                   padding: '0px 10% 0px 2%',
                   borderLeft: "3px solid rgb(69, 136, 66)"}}>
          Given vast amounts of raw acoustic data and limited annotation resources, which data should be prioritised for labelling?</p>
        <p>Active Learning (AL) is a critical strategy for scaling bioacoustic monitoring. AL is an iterative method of data selection, annotation and model training also often within a human-in-the-loop framework. Fundamentally, AL aims to optimise for a learning objective (e.g. model performance) using less labeled data minimising annotation requirements. Participants will design an AL strategy (acquisition function) to maximise training efficiency across batches of multi-label data considering informativeness quantification, diversification, long-tail performance and cross-domain generalisation.</p>

        <h3>About BioDCASE</h3>
        <p>BioDCASE (Evaluation & Benchmarking in Automated Bioacoustics) is an initiative focused on advancing research in computational bioacoustics through annual challenges and workshops. This year the <em>Active Learning for Bioacoustics</em> (Task 4) challenge will be running.</p>

        <p>Learn more about this and other challenges <a href='https://biodcase.github.io/challenge2026/summary' target=''>here</a>.</p>

        <Timeline />

        <h2>Organising Team</h2>
        <div style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '20px',
          justifyContent: 'flex-start',
          marginTop: '20px'
        }}>
          {teamMembers.map((member) => (
            <TeamMemberCard key={member.id} member={member} />
          ))}
        </div>
    </div>
  );
}