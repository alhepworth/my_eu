/* global fetch */

import postcodeAreaDataPath from './data/map/output/postcode_areas.data.json'
import { extractPostcodeArea, convertSplitRowsToRecords } from './utilities'

function parseStartAndEndDates(record) {
  const MS_PER_S = 1e3
  record.start_date = new Date(record.start_date * MS_PER_S)
  record.end_date = new Date(record.end_date * MS_PER_S)
  return record
}

export default class PostcodeAreaStore {
  constructor() {
    this.data = null
  }

  lookup(postcodeArea) {
    if (!this.data) return null

    const totalAmounts = this._lookupTotalAmounts(postcodeArea)
    if (!totalAmounts) throw new Error('postcode area not found')

    const projects = this._lookupProjects(postcodeArea)
    const cap = this._lookupCap(postcodeArea)
    const cordis = this._lookupTopProjects(postcodeArea, 'cordis')
    const creative = this._lookupTopProjects(postcodeArea, 'creative')
    const erasmus = this._lookupTopProjects(postcodeArea, 'erasmus')
    const esif = this._lookupTopProjects(postcodeArea, 'esif')

    return {
      postcodeArea,
      totalAmounts,
      projects,
      cap,
      cordis,
      creative,
      erasmus,
      esif
    }
  }

  load() {
    return fetch(postcodeAreaDataPath, {
      credentials: 'same-origin'
    })
      .then(response => {
        if (response.status === 200) return response.json()
        throw new Error('postcode area data not found')
      })
      .then(data => {
        this.data = data
      })
  }

  _lookupTotalAmounts(postcodeArea) {
    const totals = this.data.totals
    const postcodeAreaIndex = totals.columns.indexOf('postcode_area')
    return convertSplitRowsToRecords(
      totals.columns,
      totals.data.filter(row => row[postcodeAreaIndex] === postcodeArea),
      postcodeAreaIndex
    )
  }

  _lookupProjects(postcodeArea) {
    const projects = this.data.projects
    const postcodeAreaIndex = projects.columns.indexOf('postcode_area')
    return convertSplitRowsToRecords(
      projects.columns,
      projects.data.filter(row => row[postcodeAreaIndex] === postcodeArea),
      postcodeAreaIndex
    )
  }

  _lookupCap(postcodeArea) {
    const cap = this.data.cap
    const postcodeAreaIndex = cap.columns.indexOf('postcode_area')
    return convertSplitRowsToRecords(
      cap.columns,
      cap.data.filter(row => row[postcodeAreaIndex] === postcodeArea),
      postcodeAreaIndex
    )
  }

  _lookupTopProjects(postcodeArea, kind) {
    const projects = this.data[kind]
    if (!projects) return []
    const postcodeIndex = projects.columns.indexOf('postcode')
    return convertSplitRowsToRecords(
      projects.columns,
      projects.data.filter(
        row => extractPostcodeArea(row[postcodeIndex]) === postcodeArea
      )
    ).map(parseStartAndEndDates)
  }
}
