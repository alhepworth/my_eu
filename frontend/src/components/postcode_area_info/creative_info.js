import React from 'react'
import PropTypes from 'prop-types'
import { Link } from 'react-router-dom'

import SharedBetween from '../info/shared_between'

import Share from '../share'

import {
  formatRoundPounds,
  formatSemiCompactPounds,
  formatYearRange,
  getPrepositionAreaName,
  indefinitePluralise
} from '../../utilities'

const TOP_N = 3

const CreativeProject = ({ project }) => {
  const yearRange = formatYearRange(project.start_date, project.end_date)
  const postcodePath = `/postcode/${project.postcode.split(/\s/).join('/')}`

  return (
    <li className="list-group-item">
      <p className="text-truncate">{project.project}</p>
      <p className="display-4">
        {formatRoundPounds(project.max_contribution_gbp)}
      </p>
      <p className="text-muted">
        {yearRange}
        <SharedBetween
          numCountries={project.num_countries}
          numOrganisations={project.num_organisations}
        />
      </p>
      <p className="text-muted">
        {project.organisation_name},{' '}
        <Link to={postcodePath}>{project.postcode}</Link>
      </p>
    </li>
  )
}

CreativeProject.propTypes = {
  project: PropTypes.object
}

const CreativeInfo = ({ postcodeArea, creative, totalAmounts, projects }) => {
  let creativeProjects = projects.find(row => row.kind === 'creative')
  if (!creativeProjects || !creativeProjects.count) return null
  const creativeCount = creativeProjects.count
  const creativeTotal = creativeProjects.total

  let topN = creativeCount > TOP_N ? `Top ${TOP_N} ` : ''

  let moreProjects = null
  if (creativeCount > TOP_N) {
    moreProjects = (
      <p>
        Browse the map to find{' '}
        {indefinitePluralise(creativeCount - TOP_N, 'more project')}{' '}
        {getPrepositionAreaName(postcodeArea)}.
      </p>
    )
  }

  const lead =
    `The EU has invested ${formatRoundPounds(creativeTotal)} to support` +
    ` ${indefinitePluralise(
      creativeCount,
      'creative project',
      4
    )} with partners ${getPrepositionAreaName(postcodeArea)}.`
  const title = 'EU Support for Culture, Creativity and the Arts'
  const emailSubject = `${title} ${getPrepositionAreaName(postcodeArea)}`

  const id = `my-eu-postcode-area-info-${postcodeArea}-creative`
  const anchor = '#' + id

  return (
    <div className="card mt-3">
      <h3 className="card-header">
        {formatSemiCompactPounds(creativeTotal)} for Culture
      </h3>
      <div className="card-body">
        <h4 className="card-title">{title}</h4>
        <p className="card-text lead">{lead}</p>
        <Share message={lead} emailSubject={emailSubject} />
        <div id={id} className="collapse">
          <h5>
            {topN}
            Creative Projects {getPrepositionAreaName(postcodeArea)}
          </h5>
          <ul className="list-group list-group-flush">
            {creative.slice(0, TOP_N).map(project => (
              <CreativeProject key={project.my_eu_id} project={project} />
            ))}
          </ul>
          {moreProjects}
        </div>
      </div>
      <div className="card-footer text-center">
        <button
          className="btn btn-link btn-block my-eu-details-toggle collapsed"
          data-toggle="collapse"
          data-target={anchor}
          aria-expanded="false"
          aria-controls={anchor}
        >
          Details
        </button>
      </div>
    </div>
  )
}

CreativeInfo.propTypes = {
  postcodeArea: PropTypes.string,
  creative: PropTypes.array,
  totalAmounts: PropTypes.array,
  projects: PropTypes.array
}

export default CreativeInfo
