export default {
    data() {
        return {
            // TODO: Upgrade to structured format:
            //     {
            //         message: e.message,
            //         variant: "danger",
            //     }
            messages: []
        };
    },
    methods: {
        getUniqueId(member) {
            return 'object_id' in member ? 'o' + member.object_id : 'm' + member.node_id;
        },
        axiosErrorHandler(error) {
            if (error === null) {
                return;
            }

            var msg;
            if (error.response) {
                msg = `${error.response.status} ${error.response.statusText}: ${
                    error.config.url
                    }`;
            } else if (error.request) {
                console.log(error.request);
                msg = "Error in request.";
            } else {
                // Something happened in setting up the request that triggered an Error
                console.log(error);
                msg = error.message;
            }
            this.messages.unshift(msg);
            console.log(error);
        },
    }
}