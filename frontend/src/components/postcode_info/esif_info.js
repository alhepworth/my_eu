import React from 'react'
import PropTypes from 'prop-types'

import Summary from './summary'

import DisplayAmount from '../info/display_amount'
import SourceBadge from '../info/source_badge'

import Share from '../share'

import {
  formatRoundPercentage,
  formatRoundPounds,
  formatYearRange
} from '../../utilities'

const EsifInfo = ({
  startDate,
  endDate,
  organisationName,
  projectTitle,
  summary,
  funds,
  euInvestment,
  projectCost
}) => {
  const yearRange = formatYearRange(startDate, endDate)
  const displayEuInvestment = formatRoundPounds(euInvestment)
  const displayPercentage = formatRoundPercentage(euInvestment / projectCost)
  const tweet = `The EU provided ${organisationName} with ${displayEuInvestment} to fund ${projectTitle}.`

  let lead
  if (!projectCost || euInvestment >= projectCost) {
    // Note: Some are overfunded. (TODO: may not be true any more; to check)
    lead = (
      <p className="lead">
        The EU provided {organisationName} with {displayEuInvestment} to fund
        this project.
      </p>
    )
  } else {
    lead = (
      <p className="lead">
        The EU provided {organisationName} with {displayEuInvestment} to fund{' '}
        {displayPercentage} of this project.
      </p>
    )
  }

  let summaryComponent
  if (summary) summaryComponent = <Summary text={summary} />

  return (
    <React.Fragment>
      <h4>
        {projectTitle} <SourceBadge source="esif" />
      </h4>
      <DisplayAmount amount={euInvestment} />
      <p className="text-muted">{yearRange}</p>
      {lead}
      {summaryComponent}
      <Share message={tweet} />
    </React.Fragment>
  )
}

EsifInfo.propTypes = {
  startDate: PropTypes.instanceOf(Date),
  endDate: PropTypes.instanceOf(Date),
  organisationName: PropTypes.string,
  projectTitle: PropTypes.string,
  summary: PropTypes.string,
  funds: PropTypes.string,
  euInvestment: PropTypes.number,
  projectCost: PropTypes.number
}

export default EsifInfo
