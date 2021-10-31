# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as python3
"""
Schafkopf implemented in Python.
"""

import enum

import random

import numpy as np

import pyspiel

class Rank(enum.IntEnum):
  Neun = 0
  König = 1
  Zehn = 2
  Ass = 3
  Unter = 4
  Ober = 5


class Suit(enum.IntEnum):
  Schellen = 0
  Herz = 1
  Gras = 2
  Eichel = 3

class CardLocation(enum.IntEnum):
  Hand0 = 0
  Hand1 = 1
  Hand2 = 2
  Hand3 = 3
  Deck = 4
  Trick = 5

class Phase(enum.IntEnum):
  Deal = 0
  Play = 1
  GameOver = 2

# Cards are sorted first by rank and then by suit
def CardSuit(card):
  return card // _NUM_RANKS

def CardRank(card):
  return card % _NUM_RANKS

RankToValue = dict([
  (0, 0), (1, 4), (2, 10), (3, 11), (4, 2), (5, 3)
])
def CardValue(card):
  return RankToValue[CardRank(card)] 


_NUM_PLAYERS = 3
_NUM_SUITS = 1
_NUM_RANKS = 6
_NUM_CARDS = _NUM_SUITS * _NUM_RANKS
_NUM_TRICKS = _NUM_CARDS / _NUM_PLAYERS
_NUM_ACTIONS = _NUM_CARDS

_GAME_TYPE = pyspiel.GameType(
    short_name="python_schafkopf",
    long_name="Python Schafkopf",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=True)
_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=_NUM_ACTIONS,
    max_chance_outcomes=_NUM_CARDS,
    num_players=_NUM_PLAYERS,
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=_NUM_CARDS)

class SchafkopfTrick:
  def __init__(self, leader = 0):
    self._cards = []
    self._leader = leader
    self._led_suit = None

  def first_card(self):
    return self._cards[0] 

  def cards_played(self):
    return len(self._cards)

  def play_card(self, card):
    # set led_suit
    if len(self._cards) == 0:
      self._led_suit = CardSuit(card)
    self._cards.append(card)

  def player_at_position(self, pos):
    return (self._leader + pos) % _NUM_PLAYERS

  def points(self):
    return sum(CardValue(c) for c in self._cards)

  def __str__(self):
    ss = f"leader:{self._leader}; "
    ss += f"led_suit:{self._led_suit}; "
    ss += f"played:{self._cards}; "
    ss += f"points:{self.points()}"
    return ss
    

class SchafkopfGame(pyspiel.Game):
  """A Python version of Schafkopf."""

  def __init__(self, params=None):
    super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return SchafkopfState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    return SchafkopfObserver(
        iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
        params)


class SchafkopfState(pyspiel.State):
  """A python version of the Schafkopf state."""

  def __init__(self, game):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)

    self._phase = Phase.Deal
    self._card_locations = [CardLocation.Deck] * _NUM_CARDS
    self._num_cards_played = 0
    self._tricks = []

    self._returns = [0.0] * _NUM_PLAYERS
    self._points = [0.0] * _NUM_PLAYERS
    self._max_points = 0
    for c in range(_NUM_CARDS):
      self._max_points += CardValue(c)

    self._next_player = pyspiel.PlayerId.CHANCE
    self._game_over = False

  # OpenSpiel (PySpiel) API functions are below. This is the standard set that
  # should be implemented by every sequential-move game with chance.

  def current_player(self):
    """Returns id of the next player to move, or TERMINAL if game is over."""
    if self._game_over:
      return pyspiel.PlayerId.TERMINAL
    elif self._phase == Phase.Deal:
      return pyspiel.PlayerId.CHANCE
    else:
      return self._next_player

  def next_player(self):
    return (self.current_player() + 1) % _NUM_PLAYERS


  def player_cards(self, player):
    return [c for c,loc in enumerate(self._card_locations) if loc == player]

  def _legal_deal_actions(self):
    return [c for c,loc in enumerate(self._card_locations) if loc == CardLocation.Deck]

  def _legal_actions(self, player):
    """Returns a list of legal actions, sorted in ascending order."""

    # return all deal actions
    if self._phase == Phase.Deal:
      return self._legal_deal_actions()

    # get all player cards
    assert player >= 0
    pcards = self.player_cards(player)

    # noone played a card yet
    if self._num_cards_played == 0:
      return pcards

    # new trick begins, can play any card
    if self._num_cards_played % _NUM_PLAYERS == 0:
      return pcards

    # check if we must follow suit
    led_suit = self._tricks[-1]._led_suit
    same_suit = []
    for c in pcards:
      if CardSuit(c) == led_suit:
        same_suit.append(c)
    return same_suit if len(same_suit) > 0 else pcards


  def winner(self, trick):
    assert len(trick._cards) == _NUM_PLAYERS
    winner = trick.player_at_position(0)
    winner_card = trick._cards[0]
    for i,c in enumerate(trick._cards):
      if CardSuit(c) == CardSuit(winner_card):
        if CardRank(c) > CardRank(winner_card):
          winner = trick.player_at_position(i)
          winner_card = c
    return winner
      

  def chance_outcomes(self):
    """Returns the possible chance outcomes and their probabilities."""
    assert self.is_chance_node()
    cards_to_deal = self._legal_deal_actions()
    if len(cards_to_deal) == 0:
      print("ERRR no deal cards")
      exit()
    p = 1 / len(cards_to_deal)
    return [(c, p) for c in cards_to_deal]


  def _apply_deal_action(self, card):
    cards_left = self._legal_deal_actions()
    receiving_player = CardLocation(len(cards_left) % _NUM_PLAYERS)
    print("receiving:", receiving_player)
    self._card_locations[card] = receiving_player

    # last card has been dealt -> start the game
    if len(cards_left) == 1:
      self._phase = Phase.Play
      # NOTE: player0 always start the first trick for now
      self._next_player = 0


  def _apply_action(self, card):
    """Applies the specified action to the state."""
    if self._phase == Phase.Deal:
      self._apply_deal_action(card)
      return

    if self._num_cards_played % _NUM_PLAYERS == 0:
      # create new trick
      self._tricks.append(SchafkopfTrick(self.current_player()))

      self._tricks[-1].play_card(card)
      self._card_locations[card] = CardLocation.Trick
      self._num_cards_played += 1
      self._next_player = self.next_player()

    else:
      #TODO: only 2player implemented
      self._tricks[-1].play_card(card)
      self._card_locations[card] = CardLocation.Trick
      self._num_cards_played += 1

      # trick finished?
      if self._num_cards_played % _NUM_PLAYERS == 0:
        # check winner
        winner = self.winner(self._tricks[-1])

        # add points
        self._points[winner] += self._tricks[-1].points()

        # check gameover
        if self._num_cards_played == _NUM_CARDS:
          self._game_over = True
          # calc zero sum returns
          for pi in range(_NUM_PLAYERS):
            self._returns[pi] = round((self._points[pi] - self._max_points // 2) / self._max_points, 3)
        else:
          self._next_player = winner
      # trick not yet finished, continue with next player
      else:
        self._next_player = self.next_player()



  def _action_to_string(self, player, action):
    """Action -> string."""
    if player == pyspiel.PlayerId.CHANCE:
      return f"D{action}"
    else:
      return f"P{player}:A{action}"

  def is_terminal(self):
    """Returns True if the game is over."""
    return self._game_over

  def returns(self):
    """Total reward for each player over the course of the game so far."""
    return self._returns

  def __str__(self):
    """String for debug purposes. No particular semantics are required."""
    s = f"phase = {self._phase}\n"
    s += f"_num_cards_played = {self._num_cards_played}\n"
    s += f"points = {self._points}\n"
    s += f"nextplayer = {self._next_player}\n"
    s += f"num_tricks = {len(self._tricks)}\n"
    if len(self._tricks) > 0:
      s += f"cur trick = {self._tricks[-1]}\n"
    return s


class SchafkopfObserver:
  """Observer, conforming to the PyObserver interface (see observation.py)."""

  def __init__(self, iig_obs_type, params):
    """Initializes an empty observation tensor."""
    if params:
      raise ValueError(f"Observation parameters not supported; passed {params}")

    # Determine which observation pieces we want to include.
    pieces = [("player", _NUM_PLAYERS, (_NUM_PLAYERS,))]
    # TODO ?
    #if iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER:
    #  pieces.append(("private_card", 3, (3,)))
    #if iig_obs_type.public_info:
    #  if iig_obs_type.perfect_recall:
    #    pieces.append(("betting", 6, (3, 2)))
    #  else:
    #    pieces.append(("pot_contribution", 2, (2,)))

    pieces.append(("private_cards", _NUM_PLAYERS * _NUM_CARDS, (_NUM_PLAYERS, _NUM_CARDS)))
    pieces.append(("solo_player", _NUM_PLAYERS, (_NUM_PLAYERS,)))

    cards_per_trick = _NUM_CARDS * _NUM_PLAYERS
    pieces.append(("cur_trick_leader", _NUM_PLAYERS, (_NUM_PLAYERS,)))
    pieces.append(("cur_trick", cards_per_trick, (_NUM_PLAYERS,_NUM_CARDS)))
    pieces.append(("prev_trick_leader",_NUM_PLAYERS, (_NUM_PLAYERS,)))
    pieces.append(("prev_trick", cards_per_trick, (_NUM_PLAYERS,_NUM_CARDS)))

    # Build the single flat tensor.
    total_size = sum(size for name, size, shape in pieces)
    self.tensor = np.zeros(total_size, np.float32)

    # Build the named & reshaped views of the bits of the flat tensor.
    self.dict = {}
    index = 0
    for name, size, shape in pieces:
      self.dict[name] = self.tensor[index:index + size].reshape(shape)
      index += size

  def set_from(self, state, player):
    """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
    self.tensor.fill(0)
    self.dict["player"][player] = 1

    if "private_cards" in self.dict:
      for p in range(_NUM_PLAYERS):
        for c,loc in enumerate(state._card_locations):
          if loc == p:
            self.dict["private_cards"][loc][c] = 1

    # TODO: not used yet
    self.dict["solo_player"][0] = 1


    if len(state._tricks) > 0:
      cur_trick = state._tricks[-1]
      self.dict["cur_trick_leader"][cur_trick._leader] = 1

      # go trough players starting w leader
      for p in range(_NUM_PLAYERS):
        if cur_trick.cards_played() > p:
          self.dict["cur_trick"][p][cur_trick._cards[p]] = 1
          
  
    if len(state._tricks) > 1:
      prev_trick = state._tricks[-2]
      self.dict["prev_trick_leader"][prev_trick._leader] = 1

      # go trough players starting w leader
      for p in range(_NUM_PLAYERS):
        if prev_trick.cards_played() > p:
          self.dict["prev_trick"][p][prev_trick._cards[p]] = 1


  def string_from(self, state, player):
    """Observation of `state` from the PoV of `player`, as a string."""
    pieces = []
    pieces.append(f"p{player}")
    if "private_cards" in self.dict:
      for c, loc in enumerate(state._card_locations):
        pieces.append(f"c{c}loc{loc.value}")

    pieces.append(f"solo_player0")
    if len(state._tricks) > 0:
      cur_trick = state._tricks[-1]
      pieces.append(f"cur_trick_leader{cur_trick._leader}")
      pieces.append(f"cur_trick_cards{cur_trick._cards}")
    if len(state._tricks) > 1:
      cur_trick = state._tricks[-2]
      pieces.append(f"prev_trick_leader{cur_trick._leader}")
      pieces.append(f"prev_trick_cards{cur_trick._cards}")

    return " ".join(str(p) for p in pieces)


# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, SchafkopfGame)
