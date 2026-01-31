(define (problem move-pot-to-sink)
  (:domain household)
  (:objects 
      pot - object
      kitchen_island sink - location
      left right - hand
  )
  (:init 
    (found pot)
    (handempty left)
    (handempty right)
    (on pot kitchen_island)
    (on sink kitchen_island)
    (clean pot)
  )
  (:goal
    (on pot sink)
  )
)

(define (problem put-box-on-couch)
  (:domain household)
  (:objects 
      box - object
      remote - object
      green_couch - location
      left right - hand
  )
  (:init 
    (found box)
    (found remote)
    (handempty left)
    (handempty right)
    (on box green_couch)
    (on remote green_couch)
    (clean box)
    (clean remote)
    (openable box)
    (openable remote)
  )
  (:goal
    (on box green_couch)
  )
)

(define (problem put-bowl-in-microwave)
  (:domain household)
  (:objects 
      bowl - object
      microwave - location
      left right - hand
  )
  (:init 
    (found bowl)
    (handempty left)
    (handempty right)
    (on bowl microwave)
    (clean bowl)
    (openable microwave)
    (closed microwave)
  )
  (:goal
    (on bowl microwave)
  )
)