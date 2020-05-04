import Vue from 'vue';

export default class EventLog {
    constructor() {
        this.log = [];
    }

    logEvent(name, extra = null, timestamp = null) {
        timestamp = timestamp || Date.now();
        const entry = { name, extra, timestamp };
        if (Vue.config.debug) {
            console.log("logEvent", entry);
        }
        this.log.push(entry);
    }
}