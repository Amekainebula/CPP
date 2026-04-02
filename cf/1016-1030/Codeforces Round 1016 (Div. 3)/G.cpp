#include <bits/stdc++.h>
#define endl "\n"
#define ll long long
#define yes cout << "YES\n";
#define no cout << "NO\n";
#define MOD 998244353
#define ff first
#define ss second
#define pb push_back
#define eb emplace_back
#define vi vector<int>
#define vll vector<ll>
#define pii pair<int, int>
#define pll pair<ll, ll>
#define all(v) v.begin(), v.end()
#define forn(i, n) for (int i = 0; i < (int)n; ++i)
#define for1(i, n) for (int i = 1; i <= (int)n; ++i)
#define forlr(i, l, r) for (int i = (int)l; i <= (int)r; ++i)
#define forrev(i, n) for (int i = (int)(n) - 1; i >= 0; --i)
#define sz(v) int(v.size())

using namespace std;

int main(void)
{
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);
    cout << fixed;
    cout.precision(6);

    int T = 1;
    cin >> T;
    while (T--)
    {
        ll n, q, x, y, d, cnt, now;
        string op;
        cin >> n >> q;
        while (q--)
        {
            cin >> op;
            if (op[0] == '-')
            {
                cin >> x >> y;
                d = 1;
                cnt = 1;
                x--;
                y--;
                while (x > 0 || y > 0)
                {
                    if (x % 2 && y % 2)
                        d += cnt;
                    else if (x % 2)
                        d += cnt * 2ll;
                    else if (y % 2)
                        d += cnt * 3ll;
                    cnt *= 4;
                    x /= 2;
                    y /= 2;
                }
                // cout << d << "? " << x << " " << y << ": ";
                if (x == 1 && y == 1)
                    d += 1;
                else if (x == 1)
                    d += 2;
                else if (y == 1)
                    d += 3;
                cout << d << "\n";
            }
            else
            {
                cin >> d;
                x = 1, y = 1;
                cnt = 1;
                d--;
                while (d > 0)
                {
                    now = d % 4;
                    if (now == 1 || now == 2)
                        x += cnt;
                    if (now == 1 || now == 3)
                        y += cnt;
                    cnt *= 2;
                    d /= 4;
                }
                cout << x << " " << y << "\n";
            }
        }
    }

    return 0;
}