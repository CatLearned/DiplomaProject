from abc import ABC, abstractmethod


class Observer(ABC):
    """
    Интерфейс Наблюдателя объявляет метод уведомления, который издатели
    используют для оповещения своих подписчиков.
    """

    @abstractmethod
    def update_data(self, data):
        """
        Получить обновление от субъекта.
        """
        pass




