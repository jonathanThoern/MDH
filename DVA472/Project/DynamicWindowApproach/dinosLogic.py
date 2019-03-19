def detect_image(self, image, servo0):
        boxesOnLeftSide = 0
        boxesOnRightSide = 0
        linesOnLeftSide = 0
        linesOnRightSide = 0
        start = timer()
        motordrive0 = motordrive()
        #motordrive0.motadrive()
        KNOWN_WIDTH=20.0
        KNOWN_DISTANCE = 80.0
        width = 141.59543 #Default to width of the obstacle in the calibration image
        foclLenght=(width*KNOWN_DISTANCE) / KNOWN_WIDTH
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300
        
        print "Out_Classes is: ", out_classes

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            
            label = '{} {:.2f}'.format(predicted_class, score)
            if predicted_class == "Road_Line":
                lineHeight=box[2]-box[0]
                lineWidth=box[3]-box[1]
                lineCenter = box[3] - ((box[3]-box[1])/2)
                print " Line xmax", box[3], "Line ymax", box[2], "Line Center", xCenter
                if lineCenter > 208:
                    linesOnRightSide = 1
                if lineCenter <= 208:
                    linesOnLeftSide = 1
            if predicted_class == "Obstacle":
            
                height = box[2]-box[0]
                width = box[3]-box[1]
                xCenter = box[3] - ((box[3] - box[1]) /2) 
                print " xmax", box[3], "ymax", box[2], "Center", xCenter
                distanceToObstacle = (KNOWN_WIDTH * foclLenght)/width
                print "Height: " ,height, " Width: ", width, "Distance", distanceToObstacle
                if xCenter > 208 and distanceToObstacle < 80:
                    boxesOnRightSide = boxesOnRightSide+1                    
                if xCenter <= 208 and distanceToObstacle < 80:
                    boxesOnLeftSide = boxesOnLeftSide+1
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
        if(boxesOnRightSide > boxesOnLeftSide):
            servo0.controlservo(260)
            motordrive0.motordrive2times()
            servo0.resetAngle()
        elif(boxesOnRightSide < boxesOnLeftSide):
            servo0.controlservo(490)
            motordrive0.motordrive2times()
            servo0.resetAngle()
        if(linesOnLeftSide==1 and linesOnRightSide==1):
            servo0.controlservo(390)
            motordrive0.motordrive2times()
            servo0.resetAngle()
        elif(linesOnLeftSide == 1 and linesOnRightSide == 0):
            servo0.controlservo(490)
            motordrive0.motordrive2times()
            servo0.resetAngle()
        elif(linesOnRightSide == 1 and linesOnLeftSide==0):
            servo0.controlservo(260)
            motordrive0.motordrive2times()
        servo0.resetAngle()
        motordrive0.motordrive2times()
        boxesOnRightSide = 0
        boxesOnLeftSide = 0
        linesOnLeftSide = 0
        linesOnRightSide = 0
        end = timer()
        print(end - start)
        return image
